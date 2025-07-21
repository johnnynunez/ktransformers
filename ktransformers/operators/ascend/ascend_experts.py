import bisect

import acl
import torch
import numpy as np
from torch import nn
from transformers import PretrainedConfig

from ktransformers.util.ascend.ascend_utils import get_tensor_parallel_size, get_tensor_parallel_group
from ktransformers.operators.experts import KExpertsCPU, KTransformersExperts, EXPERTS_MAP, KDeepseekV3MoE, cuda_graphs
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.util.utils import CUR_DEVICE, get_use_npu_graph, InferenceState


class KExpertsCPUW8A8(KExpertsCPU):
    def __init__(
            self,
            key: str,
            gguf_loader: GGUFLoader,
            config: PretrainedConfig,
            orig_module: nn.Module = None,
            device: str = "cpu",
            **kwargs
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.input_tensor_cpu_graph = torch.zeros((1, self.config.hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16)
        self.expert_ids_cpu_graph = torch.zeros((1, self.config.num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=True)
        self.weights_cpu_graph = torch.zeros((1, self.config.num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=True)
        self.output_cpu_graph = torch.zeros((1, self.config.hidden_size), device="cpu", pin_memory=True, dtype=torch.bfloat16)
        self.bsz_tensor_cpu_graph = torch.ones((1), device="cpu", dtype=torch.int32, pin_memory=True)

    def forward(self, input_tensor, expert_ids, weights, bsz_tensor=None, cuda_graph_idx=0):
        if get_use_npu_graph():
            self.cpu_infer.submit(self.moe.forward(self.expert_ids_cpu_graph.size(0),
                                                   self.expert_ids_cpu_graph.size(1),
                                                   self.expert_ids_cpu_graph.data_ptr(),
                                                   self.weights_cpu_graph.data_ptr(),
                                                   self.input_tensor_cpu_graph.data_ptr(),
                                                   self.output_cpu_graph.data_ptr(),
                                                   self.bsz_tensor_cpu_graph.data_ptr()))
            self.cpu_infer.sync()
        else:
            if bsz_tensor is None:
                bsz_tensor = torch.tensor([input_tensor.size(0)], device=input_tensor.device, dtype=torch.int32)
            org_type = input_tensor.dtype
            input_tensor = input_tensor.contiguous().cpu()
            input_tensor = input_tensor.to(torch.bfloat16)
            expert_ids = expert_ids.contiguous().cpu()
            weights = weights.contiguous().to(torch.float32).cpu()
            bsz_tensor = bsz_tensor.contiguous().cpu()
            output = torch.empty_like(input_tensor).contiguous()
            self.cpu_infer.submit(self.moe.forward(expert_ids.size(0), expert_ids.size(1), expert_ids.data_ptr(), weights.data_ptr(), input_tensor.data_ptr(), output.data_ptr(), bsz_tensor.data_ptr()))
            self.cpu_infer.sync()
            return output.to(org_type).to(device=CUR_DEVICE)


EXPERTS_MAP["KExpertsCPUW8A8"] = KExpertsCPUW8A8


class KTransformersExpertsW8A8(KTransformersExperts):
    def forward(self, input_tensor, expert_ids, weights):
        if self.mode == InferenceState.GENERATE:
            assert self.generate_experts is not None, "generate_experts is None"
            return self.generate_experts.forward(input_tensor, expert_ids, weights)
        elif self.mode == InferenceState.PREFILL:
            assert self.prefill_experts is not None, "prefill_experts is None"
            return self.prefill_experts.forward(input_tensor, expert_ids, weights)
        else:
            raise ValueError("load or set_inference_mode before forward")


class KDeepseekV3MoEW8A8(KDeepseekV3MoE):
    def forward_tp(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        rank = torch.distributed.get_rank()
        def share_experts_forward():
            if self.config.n_shared_experts is not None:
                return self.shared_experts(identity).squeeze(0)
        if rank == 0:
            topk_idx, topk_weight = self.gate(hidden_states)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            cuda_graph_idx = bisect.bisect_left(cuda_graphs, 1)
            if get_use_npu_graph():
                from ktransformers.util.npu_graph_runner import get_or_create_runner
                npu_graph_runner = get_or_create_runner(CUR_DEVICE)
                event = torch.npu.Event()
                event.record(npu_graph_runner.main_stream)
                with torch.npu.stream(npu_graph_runner.update_stream):
                    event.wait(npu_graph_runner.update_stream)
                    y_ = share_experts_forward() if share_experts_forward is not None else None
                    event.record(npu_graph_runner.update_stream)
                org_type = hidden_states.dtype
                input_tensor = hidden_states.to(torch.bfloat16)
                topk_weight = topk_weight.contiguous().to(torch.float32)
                self.moe_kexperts_param = (hidden_states, topk_idx, topk_weight)
                self.experts.generate_experts.input_tensor_cpu_graph.copy_(input_tensor, non_blocking=True)
                self.experts.generate_experts.expert_ids_cpu_graph.copy_(topk_idx, non_blocking=True)
                self.experts.generate_experts.weights_cpu_graph.copy_(topk_weight, non_blocking=True)

                npu_graph_runner.launch_callback(
                    self.cpu_moe_kexperts,
                    self.moe_kexperts_param,
                    1, npu_graph_runner.stream)

                output_npu_graph = self.experts.generate_experts.output_cpu_graph.to(CUR_DEVICE, non_blocking=True)
                y = output_npu_graph.to(org_type)
                event.wait(npu_graph_runner.main_stream)
            else:
                y = self.moe_kexperts(hidden_states, topk_idx, topk_weight)
                y_ = share_experts_forward() if share_experts_forward is not None else None
                y = y.view(*orig_shape).to(device=hidden_states.device)
        else:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            y = torch.zeros(orig_shape, dtype=torch.float16, device=CUR_DEVICE)
            y_ = share_experts_forward() if share_experts_forward is not None else None
        torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM, group=get_tensor_parallel_group())
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    def forward(self, hidden_states):
        tp_size = get_tensor_parallel_size()
        world_size = torch.distributed.get_world_size()
        if tp_size > 1 and world_size == tp_size:
            return self.forward_tp(hidden_states)
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y_ = None

        # only for generate phase
        # if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_current_stream_capturing():
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode") and False:
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], topk_idx[0], topk_weight[0])
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity).squeeze(0)
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y

        def share_experts_forward():
            if self.config.n_shared_experts is not None:
                return self.shared_experts(identity).squeeze(0)

        cuda_graph_idx = bisect.bisect_left(cuda_graphs, 1)
        if get_use_npu_graph():
            from ktransformers.util.npu_graph_runner import get_or_create_runner
            npu_graph_runner = get_or_create_runner(CUR_DEVICE)
            event = torch.npu.Event()
            event.record(npu_graph_runner.main_stream)
            with torch.npu.stream(npu_graph_runner.update_stream):
                event.wait(npu_graph_runner.update_stream)
                y_ = share_experts_forward() if share_experts_forward is not None else None
                event.record(npu_graph_runner.update_stream)
            org_type = hidden_states.dtype
            input_tensor = hidden_states.to(torch.bfloat16)
            topk_weight = topk_weight.contiguous().to(torch.float32)
            self.moe_kexperts_param = (hidden_states, topk_idx, topk_weight)
            self.experts.generate_experts.input_tensor_cpu_graph.copy_(input_tensor, non_blocking=True)
            self.experts.generate_experts.expert_ids_cpu_graph.copy_(topk_idx, non_blocking=True)
            self.experts.generate_experts.weights_cpu_graph.copy_(topk_weight, non_blocking=True)

            npu_graph_runner.launch_callback(
                self.cpu_moe_kexperts,
                self.moe_kexperts_param,
                1, npu_graph_runner.stream)

            output_npu_graph = self.experts.generate_experts.output_cpu_graph.to(CUR_DEVICE, non_blocking=True)
            y = output_npu_graph.to(org_type)
            event.wait(npu_graph_runner.main_stream)
        else:
            y = self.moe_kexperts(hidden_states, topk_idx, topk_weight)
            y_ = share_experts_forward() if share_experts_forward is not None else None
            y = y.view(*orig_shape).to(device=hidden_states.device)

        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @torch.no_grad()
    def cpu_moe_kexperts(self, moe_kexperts_param) -> torch.Tensor:
        x, topk_ids, topk_weight = moe_kexperts_param
        self.moe_kexperts(x, topk_ids, topk_weight)

    @torch.no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs