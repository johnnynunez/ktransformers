'''
Description :
Author      : Boxin Zhang
Version     : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
'''
from typing import Dict

import acl
import torch
import torch_npu
from torch import nn

import ktransformers.util.npu_graph as npu_graph
from ktransformers.util.utils import CUR_DEVICE


class NPUGraphRunner:
    def __init__(self, deviceId):
        torch.npu.set_compile_mode(jit_compile=False)
        self.deviceId = deviceId
        self.enable = False
        self.debug = False
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}
        self.tid = None
        self.past_key_value = None

    def init(self, batch_size, seq_length):
        self.tmp_g = npu_graph.NpuGraph()
        self.graph = torch.npu.NPUGraph()
        self.main_stream = torch_npu.npu.Stream(device=self.deviceId)
        self.update_stream = torch_npu.npu.Stream(device=self.deviceId)
        self.stream = self.main_stream.npu_stream
        self.logits = torch.zeros((batch_size, seq_length, 7168), dtype=torch.float16).to(self.deviceId)
        self.context, ret = acl.rt.get_context(self.deviceId)
        if ret != 0:
            print("get_context failed! ret: " + str(ret))
            exit(-1)
        self.exit_flag = False
        self.handle = []
        self.ifa_param = []
        self.event = []
        self.first_update = True
        self.workspace = None

        if self.tid is None:
            def process_callback(args_list):
                ins = args_list[0]
                ret = acl.rt.set_context(ins.context)
                if ret != 0:
                    print("set_context failed! ret: " + str(ret))
                    exit(-1)

                while True:
                    acl.rt.process_report(1)
                    if ins.exit_flag:
                        break

            self.tid, ret = acl.util.start_thread(process_callback, [self])
            if ret != 0:
                print("start_thread failed!")
                exit(-1)

        ret = acl.rt.subscribe_report(self.tid, self.stream)
        if ret != 0:
            print("subscribe_report failed!")
            exit(-1)

    def destroy(self):
        print(f'[rank:{torch.distributed.get_rank()}]------------- NPU Graph Destroy Begin -------------\n', end='')
        self.exit_flag = True
        ret = acl.rt.unsubscribe_report(self.tid, self.stream)
        if ret != 0:
            print("unsubscribe_report failed!")
            exit(-1)
        self.enable = False
        ret = acl.util.stop_thread(self.tid)
        if ret != 0:
            print("stop_thread failed!")
            exit(-1)
        self.tid = None
        self.workspace = None
        self.handle = []
        self.ifa_param = []
        self.event = []
        self.first_update = True
        del self.graph
        self.tmp_g.destroy()
        destroy_runner(self.deviceId)
        print(f'[rank:{torch.distributed.get_rank()}]------------- NPU Graph Destroy Finish -------------\n', end='')

    def capture(
            self,
            model,
            cur_token,
            position_ids,
            cache_position,
            past_key_values,
            main_device,
            **kwargs,
    ) -> None:
        print(f'[rank:{torch.distributed.get_rank()}]------------- NPU Graph Capture Begin -------------\n', end='')
        self.enable = True
        self.model = model
        inputs_embeds = model.model.embed_tokens(cur_token.to("cpu")).to(main_device)
        self.seq_length = inputs_embeds.size()[1]
        self.main_device = main_device
        with torch.no_grad():
            with torch.npu.graph(self.graph, stream=self.main_stream):
                self.logits = model(inputs_embeds=inputs_embeds,
                                    position_ids=position_ids,
                                    cache_position=cache_position,
                                    past_key_values=past_key_values,
                                    **kwargs)[0]

        if past_key_values != None:
            past_key_values.change_seq_length(-1)

        self.input_buffers = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "cache_position": cache_position,
        }

        self.output_buffers = {"logits": self.logits}
        print(f'[rank:{torch.distributed.get_rank()}]------------- NPU Graph Capture Finish -------------\n', end='')
        return

    def forward(
            self,
            inputs_embeds,
            position_ids,
            cache_position,
    ) -> torch.Tensor:
        def ifa_update_sync(param):
            with torch.npu.stream(self.update_stream):
                for i in range(len(self.handle)):
                    if self.first_update is False:
                        q_nope, kvCache, q_pe, kRopeCache, num_heads, \
                            softmax_scale, layer_idx, attn_output, softmax_lse = self.ifa_param[i]
                        torch.npu.graph_task_update_begin(self.update_stream, self.handle[i])
                        torch_npu.npu_fused_infer_attention_score.out(
                            q_nope,
                            kvCache,
                            kvCache,
                            workspace=self.workspace,
                            query_rope=q_pe,
                            key_rope=kRopeCache,
                            num_heads=num_heads,
                            num_key_value_heads=1,
                            input_layout="BNSD",
                            atten_mask=None,
                            scale=softmax_scale,
                            antiquant_mode=0,
                            antiquant_scale=None,
                            block_table=self.past_key_value.page_table_list[layer_idx],
                            block_size=self.past_key_value.page_size,
                            actual_seq_lengths_kv=self.past_key_value.position,
                            out=[attn_output, softmax_lse])
                        torch.npu.graph_task_update_end(self.update_stream)
                    self.event[i].record(self.update_stream)

        self.ifa_update_tid, ret = acl.util.start_thread(ifa_update_sync, [self])
        if ret != 0:
            print("start_thread failed!")
            exit(-1)

        ret1 = acl.rt.memcpy(self.input_buffers["inputs_embeds"].data_ptr(), inputs_embeds.numel() * 2,
                             inputs_embeds.data_ptr(), inputs_embeds.numel() * 2, 3)
        ret2 = acl.rt.memcpy(self.input_buffers["position_ids"].data_ptr(), position_ids.numel() * 8,
                             position_ids.data_ptr(), position_ids.numel() * 8, 3)
        ret3 = acl.rt.memcpy(self.input_buffers["cache_position"].data_ptr(), cache_position.numel() * 8,
                             cache_position.data_ptr(), cache_position.numel() * 8, 3)
        torch_npu.npu.synchronize()

        with torch_npu.npu.stream(self.main_stream):
            self.graph.replay()
        self.first_update = False
        ret = acl.util.stop_thread(self.ifa_update_tid)
        if ret != 0:
            print("stop_thread failed!")
            exit(-1)
        else:
            self.ifa_update_tid = None
        return self.output_buffers["logits"]

    def launch_callback(self, func, data, block, stream):
        self.tmp_g.launch_callback(func, data, block, stream)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


runner_dict = dict()


def check_runner(deviceId: int):
    runner = runner_dict.get(deviceId)
    if runner is None:
        return True
    else:
        return False


def destroy_runner(deviceId: int):
    runner = runner_dict.get(deviceId)
    if runner is not None:
        runner_dict[deviceId] = None


def get_or_create_runner(deviceId: int):
    runner = runner_dict.get(deviceId)

    if runner is None:
        runner = NPUGraphRunner(deviceId)
        runner_dict[deviceId] = runner
    return runner