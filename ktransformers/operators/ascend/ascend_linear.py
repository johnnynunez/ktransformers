from abc import abstractmethod

import torch
import torch_npu
import torch.distributed as dist
from torch import nn
from transformers import PretrainedConfig

from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.operators.linear import KLinearBase, LINEAR_MAP
from ktransformers.util.ascend.ascend_utils import (
    get_safetensors_cut_weight,
    get_tensor_parallel_size,
    get_tensor_parallel_group
)
from ktransformers.util import utils
from ktransformers.util.custom_loader import GGUFLoader
from ktransformers.util.utils import InferenceState
from ktransformers.util.custom_loader import translate_name_to_gguf

class KLinearW8A8(KLinearBase):
    def __init__(
            self,
            key: str,
            gguf_loader: GGUFLoader,
            config: PretrainedConfig,
            orig_module: nn.Module = None,
            device: str = "cuda",
            **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)

    def load_weight(self, override_key: str | None = None, device: str | None = None):
        if override_key is not None:
            keys = override_key
        else:
            keys = [self.key]
        fake_tensor = torch.tensor([1])
        for key in keys:
            if device is None:
                device = utils.CUR_DEVICE
            
            key = translate_name_to_gguf(key)
            if key == "lm_head":
                key = "output"

            if key + ".weight" in self.gguf_loader.safetensor_loader.tensor_file_map:
                if key + ".deq_scale" in self.gguf_loader.safetensor_loader.tensor_file_map:
                    qweight = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.weight")
                    deq_scale = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.deq_scale")
                    quant_bias = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.quant_bias")
                    input_scale = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.input_scale")
                    input_offset = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.input_offset")
                    tensors = (qweight, deq_scale, quant_bias, input_scale, input_offset)
                    print(f"Loading {key} with shape {qweight.shape}, {deq_scale.shape}, {quant_bias.shape}, {input_scale.shape}, {input_offset.shape}")
                    print(tensors)
                    return tensors
                elif key + ".weight_scale" in self.gguf_loader.safetensor_loader.tensor_file_map:
                    if key.endswith("ffn_gate_shexp"):
                        parts = key.split(".")
                        layer = parts[1]
                        gate_weight = self.gguf_loader.safetensor_loader.load_tensor(f"blk.{layer}.ffn_gate_shexp.weight")
                        up_weight = self.gguf_loader.safetensor_loader.load_tensor(f"blk.{layer}.ffn_up_shexp.weight")
                        gate_up_weight = torch.cat((gate_weight, up_weight), 0)
                        gate_scale = self.gguf_loader.safetensor_loader.load_tensor(f"blk.{layer}.ffn_gate_shexp.weight_scale")
                        up_scale = self.gguf_loader.safetensor_loader.load_tensor(f"blk.{layer}.ffn_up_shexp.weight_scale")
                        gate_up_scale = torch.cat((gate_scale, up_scale), 0)
                        gate_offset = self.gguf_loader.safetensor_loader.load_tensor(f"blk.{layer}.ffn_gate_shexp.weight_offset")
                        up_offset = self.gguf_loader.safetensor_loader.load_tensor(f"blk.{layer}.ffn_up_shexp.weight_offset")
                        gate_up_offset = torch.cat((gate_offset, up_offset), 0)
                        tensors = (gate_up_weight, gate_up_scale, gate_up_offset)
                        print(f"Loading {key} as ffn_gate_shexp with shape {gate_up_weight.shape}, {gate_up_scale.shape}, {gate_up_offset.shape}")
                        print(tensors)
                    elif key.endswith("ffn_up_shexp"):
                        return fake_tensor
                    else:
                        qweight = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.weight")
                        weight_scale = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.weight_scale")
                        weight_offset = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.weight_offset")
                        tensors = (qweight, weight_scale, weight_offset)
                        print(f"Loading {key} with shape {qweight.shape}, {weight_scale.shape}, {weight_offset.shape}")
                        print(tensors)
                    return tensors
                else:
                    weight = self.gguf_loader.safetensor_loader.load_tensor(f"{key}.weight")
                    return weight
            else:
                raise FileNotFoundError(f"Weight file not found for key {key}")

    @abstractmethod
    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str | None = "cuda"):
        pass

    @abstractmethod
    def unload(self):
        pass


class KLinearTorchW8A8A2(KLinearW8A8):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, device, **kwargs)
        self.has_bias = False
        self.dtype = torch.get_default_dtype()
        self.weight = None
        self.input_scale = None
        self.input_offset = None
        self.quant_bias = None
        self.deq_scale = None
        self.weight_scale = None
        self.weight_offset = None

    def forward(self, x: torch.Tensor, bsz_tensor) -> torch.Tensor:
        tp = get_tensor_parallel_size()
        if tp == 1:
            out = torch.zeros((x.shape[0], x.shape[1], self.weight.shape[-1]), dtype=torch.float16, device=x.device)
            torch_npu._npu_matmul_pp(x, self.weight, out)
        else:
            tp_size = get_tensor_parallel_size()
            tp_group = get_tensor_parallel_group()
            batch_size = x.shape[0]
            seq_length = x.shape[1]
            lm_sep_size = tp_size
            lm_head_group = tp_group
            gathered_list = [torch.empty_like(x) for _ in range(lm_sep_size)]
            dist.all_gather(gathered_list, x, group=lm_head_group)
            input_full = torch.stack(gathered_list, dim=0)
            input_full = input_full.squeeze(dim=1)
            torch_npu.npu_format_cast_(input_full, 2)
            local_logits = torch.zeros((input_full.shape[0], input_full.shape[1], self.weight.shape[-1]),
                                       dtype=torch.float16, device=input_full.device)
            torch_npu._npu_matmul_pp(input_full, self.weight, local_logits)
            local_logits_transpose = local_logits.transpose(2, 1).reshape(-1, batch_size * seq_length)
            del local_logits
            output_tensor = torch.empty_like(local_logits_transpose)
            dist.all_to_all_single(output_tensor, local_logits_transpose, group=lm_head_group)
            del local_logits_transpose
            output_tensor = output_tensor.transpose(1, 0)
            out = output_tensor.view(batch_size, seq_length, -1)
            del output_tensor
        return out

    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str | None = None):
        if device is None:
            device = self.device
        device = utils.CUR_DEVICE
        if w is None:
            w = self.load_weight()
        if isinstance(w, nn.Parameter):
            try:
                self.weight = w.to(dtype=self.dtype).view(self.out_features, self.in_features).T.contiguous()
            except:
                self.weight = w.to(dtype=self.dtype).T.contiguous()
            self.weight = self.weight.to(device)
            if self.has_bias:
                self.bias = self.bias.to(device)
        elif isinstance(w, tuple):
            w_list = list(w)
            if len(w_list) == 3:
                self.weight = w_list[0]
                self.weight_scale = w_list[1].view(-1)
                self.weight_offset = w_list[2]
                self.weight = self.weight.to(utils.CUR_DEVICE)
                self.weight_scale = self.weight_scale.to(utils.CUR_DEVICE)
                if self.key.endswith("ffn_gate_shexp") is not True:
                    self.weight = get_safetensors_cut_weight(self.key, self.weight).t()
                    weight_scale = get_safetensors_cut_weight(self.key, self.weight_scale)
                    self.weight_scale = weight_scale.clone()
                    del weight_scale
                self.weight_offset = self.weight_offset.to(utils.CUR_DEVICE)
            else:
                for i in range(len(w_list)):
                    w_list[i] = get_safetensors_cut_weight(self.key, w_list[i])
                    w_list[i] = w_list[i].to(utils.CUR_DEVICE)
                self.weight = w_list[0]
                self.deq_scale = w_list[1]
                self.quant_bias = w_list[2]
                if "attn_output" in self.key or "ffn_down" in self.key:
                    if torch.distributed.get_rank(get_tensor_parallel_group()) != 0:
                        self.quant_bias = torch.zeros_like(self.quant_bias, dtype=self.quant_bias.dtype, device=self.quant_bias.device)
                self.input_scale = w_list[3]
                self.input_offset = w_list[4]
        elif isinstance(w, torch.Tensor):
            self.weight = w.T.contiguous()
            self.weight.to(device)
            if "kv_b" not in self.key:
                self.weight = self.weight.to(device)
                torch_npu.npu_format_cast_(self.weight, 29)
        else:
            raise ValueError(f"Invalid weight type {self.key=} {type(w)=}")

    def unload(self):
        if self.weight is not None:
            self.weight = None
        if self.has_bias:
            self.bias = None
        self.input_scale = None
        self.input_offset = None
        self.quant_bias = None
        self.deq_scale = None
        self.weight_scale = None
        self.weight_offset = None


LINEAR_MAP["KLinearTorchW8A8A2"] = KLinearTorchW8A8A2


class KTransformersLinearW8A8A2(BaseInjectedModule, KLinearW8A8):
    def __init__(
            self,
            key: str,
            gguf_loader: GGUFLoader,
            config: PretrainedConfig,
            orig_module: nn.Module,
            generate_device: str = "cuda",
            generate_op: str | None = "KLinearMarlin",
            prefill_device: str = "cuda",
            prefill_op: str | None = "KLinearTorch",
            **kwargs,
    ):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device, **kwargs)
        KLinearW8A8.__init__(self, key, gguf_loader, config, orig_module, generate_device, **kwargs)
        # build all the linear operators
        if prefill_op is not None:
            assert prefill_op in LINEAR_MAP, f"linear_type {prefill_op} not supported"
            self.prefill_linear = LINEAR_MAP[prefill_op](key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        else:
            self.prefill_linear = None

        if generate_op is not None:
            assert generate_op in LINEAR_MAP, f"linear_type {generate_op} not supported"
            self.generate_linear = LINEAR_MAP[generate_op](key, gguf_loader, config, orig_module, generate_device, **kwargs)
        else:
            self.generate_linear = None
        self.mode = InferenceState.UNLOAD

    def forward(self, x, bsz_tensor=None):
        if self.mode == InferenceState.PREFILL:
            assert self.prefill_linear is not None, "cpu linear is not initialized"
            y = self.prefill_linear.forward(x, bsz_tensor)
        else:
            assert self.generate_linear is not None, "gpu linear is not initialized"
            y = self.generate_linear.forward(x, bsz_tensor)
        return y

    def load(self, w: dict | nn.Parameter | tuple | None = None, mode: InferenceState = InferenceState.GENERATE):
        if not mode:
            mode = InferenceState.GENERATE
        # load to device
        if mode == InferenceState.PREFILL:
            self.generate_linear.unload()
            self.prefill_linear.load(w=w)
            self.device = self.prefill_linear.device
            self.weight = self.prefill_linear.weight  # modeling_xxx.py may use linear.weight
            self.input_scale = self.prefill_linear.input_scale
            self.input_offset = self.prefill_linear.input_offset
            self.quant_bias = self.prefill_linear.quant_bias
            self.deq_scale = self.prefill_linear.deq_scale
            self.weight_scale = self.prefill_linear.weight_scale
            self.weight_offset = self.prefill_linear.weight_offset
        elif mode == InferenceState.GENERATE:
            self.prefill_linear.unload()
            self.generate_linear.load(w=w)
            self.device = self.generate_linear.device
            self.weight = self.generate_linear.weight  # modeling_xxx.py may use linear.weight
            self.input_scale = self.generate_linear.input_scale
            self.input_offset = self.generate_linear.input_offset
            self.quant_bias = self.generate_linear.quant_bias
            self.deq_scale = self.generate_linear.deq_scale
            self.weight_scale = self.generate_linear.weight_scale
            self.weight_offset = self.generate_linear.weight_offset
        elif mode == InferenceState.UNLOAD:
            self.prefill_linear.unload()
            self.generate_linear.unload()
            self.device = "cpu"
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")
        self.mode = mode

    def unload(self):
        if self.prefill_linear is not None:
            self.prefill_linear.unload()
        if self.generate_linear is not None:
            self.generate_linear.unload()
        self.device = self.generate_linear.device

    def set_inference_mode(self, mode: InferenceState):
        if not mode:
            mode = InferenceState.GENERATE
        if mode == InferenceState.GENERATE:
            self.load(mode=InferenceState.GENERATE)
        elif mode == InferenceState.PREFILL:
            self.load(mode=InferenceState.PREFILL)
        elif mode == InferenceState.UNLOAD:
            self.unload()
        else:
            raise ValueError("mode must be either InferenceState.GENERATE, InferenceState.PREFILL or InferenceState.UNLOAD")