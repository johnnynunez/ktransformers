import os
import warnings
from typing import Optional, Tuple

import torch
import torch_npu
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import Cache

from ktransformers.models.modeling_deepseek import DeepseekV2Attention, apply_rotary_pos_emb
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.util.ascend.ascend_utils import get_tensor_parallel_size, allreduce_wrapper
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.util.utils import get_compute_capability, get_use_npu_graph, CUR_DEVICE
from ktransformers.util.vendors import device_manager, GPUVendor
from ktransformers.util import utils


def apply_rotary_pos_emb_fusion(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


class MatMulOps(object):
    def execute(self, x_input):
        """
            :param x, weight, quant_bia, deq_scale
            :return:
        """
        quant_out = x_input[0]
        weight = x_input[1]
        quant_bia = x_input[2]
        deq_scale = x_input[3]
        return [torch_npu.npu_quant_matmul(quant_out, weight.T, deq_scale, bias=quant_bia, output_dtype=torch.float16)]


class MatMulOpsAtb(object):
    def execute(self, x_input):
        """
            :param x, weight, quant_bia, deq_scale
            :return:
        """
        x = x_input[0]
        weight = x_input[1]
        quant_bia = x_input[2]
        deq_scale = x_input[3]
        target_shape = (x.shape[0], x.shape[-2], weight.shape[-2])
        target_tensor = torch.zeros(target_shape, dtype=torch.float16, device=x.device)
        torch_npu.torch_npu._npu_matmul_dequant(x, weight, quant_bia, deq_scale, target_tensor)
        return [target_tensor]


class DynamicQuantOps(object):
    """
        :param x, scale, offset
        :return
    """

    def execute(self, x_input):
        out = torch.empty_like(x_input[0], dtype=torch.int8)
        torch_npu._npu_quantize_per_tensor(x_input[0], x_input[1], x_input[2], out)
        return [out]


class KDeepseekV2AttentionW8A8A2(BaseInjectedModule, DeepseekV2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    attn_mask: Optional[torch.Tensor] = None

    def __init__(self,
                 key: str,
                 gguf_loader: GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "cuda",
                 generate_device: str = "cuda",
                 chunck_size: int = 1000,
                 absorb_for_prefill: bool = False,
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, generate_device,
                                    **kwargs)
        self.orig_module.__init__(orig_module.config,
                                  orig_module.layer_idx)
        self.chunck_size = chunck_size  # TODO, generate chunck_size automatically.
        self.mla_wrapper = None
        tp = get_tensor_parallel_size()
        if tp > 1:
            self.num_heads //= tp
        self.absorb_for_prefill = absorb_for_prefill

        self.use_merge = os.getenv("USE_MERGE", "0")
        if self.use_merge == "0":
            print("--Use ATB FA-MLA and PA-MLA OP !--")
            self.elewise_quant = DynamicQuantOps()
            self.matmulDequant_operation = MatMulOpsAtb()
            self.matmulDequant_operation_aclnn = MatMulOps()
        elif self.use_merge == "1":
            print("--Use torch npu FA OP !--")
        else:
            print("--Use default op! --")

    @allreduce_wrapper
    def forward_chunck(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            hidden_states_quant = self.elewise_quant.execute([hidden_states, self.q_a_proj.input_scale, self.q_a_proj.input_offset])[0]
            q_a_proj_out = self.matmulDequant_operation.execute([hidden_states_quant, self.q_a_proj.weight,
                                                                 self.q_a_proj.quant_bias, self.q_a_proj.deq_scale])[0]
            q_a_proj_out = self.q_a_layernorm(q_a_proj_out)
            q_a_proj_out = self.elewise_quant.execute([q_a_proj_out, self.q_b_proj.input_scale, self.q_b_proj.input_offset])[0]
            q = self.matmulDequant_operation.execute([q_a_proj_out, self.q_b_proj.weight,
                                                      self.q_b_proj.quant_bias, self.q_b_proj.deq_scale])[0]
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        hidden_states_quant = self.elewise_quant.execute([hidden_states, self.kv_a_proj_with_mqa.input_scale, self.kv_a_proj_with_mqa.input_offset])[0]
        compressed_kv = self.matmulDequant_operation.execute([hidden_states_quant, self.kv_a_proj_with_mqa.weight,
                                                              self.kv_a_proj_with_mqa.quant_bias, self.kv_a_proj_with_mqa.deq_scale])[0]
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv_seq_len = k_pe.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since transformer version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(q_pe, position_ids)
        q_pe, k_pe = apply_rotary_pos_emb_fusion(q_pe, k_pe, cos, sin)

        # update KV
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            k_pe = k_pe.transpose(1, 2)                 # k_pe [bsz, 1, q_len, self.qk_rope_head_dim]
            compressed_kv = compressed_kv.unsqueeze(2)  # compressed_kv [bsz, q_len, self.kv_lora_rank]
            compressed_kv_with_k_pe, _ = past_key_value.update(compressed_kv, k_pe, self.layer_idx, cache_kwargs)
            compressed_kv, k_pe = torch.split(
                compressed_kv_with_k_pe, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        k_pe = k_pe.view(bsz, 1, -1, self.qk_rope_head_dim)[:, :, :attention_mask.size(-1), :]
        compressed_kv = compressed_kv.view(bsz, 1, -1, self.kv_lora_rank)[:, :, :attention_mask.size(-1), :]

        weight_uk = self.q_absorb
        weight_uv = self.out_absorb

        # ATB-MLA-FA+PA
        if self.use_merge == "0" and q_len != 1:
            current_sqenLen = past_key_value.get_seq_length(self.layer_idx)
            attention_mask = attention_mask[0, :, :, :current_sqenLen].squeeze(0).squeeze(0)

            compressed_kv = compressed_kv[:, :, :current_sqenLen, :]  # all KV until current chunk
            k_pe = k_pe[:, :, :current_sqenLen, :]

            k_pe_repeated = k_pe.repeat(1, self.num_heads, 1, 1)
            k_up = torch.matmul(compressed_kv, weight_uk.mT)
            v_up = torch.matmul(compressed_kv, weight_uv)

            qTensor = torch.cat((q_nope, q_pe), dim=-1).transpose(1, 2).contiguous().view(
                bsz * q_len, self.num_heads, (self.qk_nope_head_dim + self.qk_rope_head_dim))
            kTensor = torch.cat((k_up, k_pe_repeated), dim=-1).transpose(1, 2).contiguous().view(
                bsz * current_sqenLen, self.num_heads, (self.qk_nope_head_dim + self.qk_rope_head_dim))
            vTensor = v_up.transpose(1, 2).contiguous().view(bsz * current_sqenLen, self.num_heads, self.v_head_dim)

            seq_len_data = [q_len] * bsz
            seq_len = torch.tensor(seq_len_data, dtype=torch.int32, device=vTensor.device)
            seq_len_host = torch.tensor(seq_len_data, dtype=torch.int32)

            attn_output = torch.ones((qTensor.shape[0], qTensor.shape[1], vTensor.shape[-1]),
                                     dtype=qTensor.dtype, device=vTensor.device)
            torch_npu._npu_flash_attention_mla(qTensor, kTensor, vTensor, attention_mask, seq_len, seq_len_host,
                                               self.softmax_scale, self.num_heads, self.num_heads, attn_output)

            if attn_output.size() != (bsz * q_len, self.num_heads, self.v_head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, q_len, self.num_heads, self.v_head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.view(bsz, q_len, self.num_heads * self.v_head_dim)
            attn_output = self.elewise_quant.execute([attn_output, self.o_proj.input_scale, self.o_proj.input_offset])[0]
            attn_output = self.matmulDequant_operation_aclnn.execute([attn_output, self.o_proj.weight,
                                                                     self.o_proj.quant_bias, self.o_proj.deq_scale])[0]

            return attn_output, None, past_key_value

        elif self.use_merge == "0" and q_len == 1:
            return self.forward_paged(q_pe=q_pe,
                                      q_nope=q_nope,
                                      compressed_kv_with_k_pe=compressed_kv_with_k_pe,
                                      past_key_value=past_key_value,
                                      cache_position=cache_position)

        if self.use_merge == "1":
            k_pe_repeated = k_pe.repeat(1, self.num_heads, 1, 1)
            k_up = torch.matmul(compressed_kv, weight_uk.mT)
            v_up = torch.matmul(compressed_kv, weight_uv)
            qTensor = torch.cat((q_nope, q_pe), dim=-1)
            kTensor = torch.cat((k_up, k_pe_repeated), dim=-1)
            vTensor = torch.cat((v_up, k_pe_repeated), dim=-1)

            if q_len != 1:
                attn_output = torch_npu.npu_prompt_flash_attention(
                    qTensor, kTensor, vTensor,
                    num_heads=self.num_heads, scale_value=self.softmax_scale, input_layout="BNSD")
            else:
                attn_output = torch_npu.npu_incre_flash_attention(
                    qTensor, kTensor, vTensor,
                    num_heads=self.num_heads, scale_value=self.softmax_scale, input_layout="BNSD")
            attn_output = attn_output[:, :, :, :self.v_head_dim]
        else:
            q_nope = torch.matmul(q_nope, self.q_absorb)

            attn_weights = (torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.mT)) * self.softmax_scale

            compressed_kv = compressed_kv.squeeze(1)
            """
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )
            assert attention_mask is not None
            """
        if attention_mask is not None:
            """
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            """
            attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q_pe.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.attention_dropout, training=self.training
            )
            attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)

            attn_output = torch.matmul(attn_output, self.out_absorb)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def forward_paged(
        self,
        q_pe: torch.Tensor,
        q_nope: torch.Tensor,
        compressed_kv_with_k_pe: torch.Tensor,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, _, q_len, _ = q_nope.size()
        q_nope = torch.einsum('b h q d, h d k -> b h q k', q_nope, self.q_absorb)  # torch.Size([1, 128, 1, 512])
        compressed_kv = compressed_kv_with_k_pe.permute(0, 2, 1, 3)
        kvCache = compressed_kv[:, :, :, :self.kv_lora_rank].contiguous()
        kRopeCache = compressed_kv[:, :, :, self.kv_lora_rank:].contiguous()

        if get_use_npu_graph():
            from ktransformers.util.npu_graph_runner import get_or_create_runner
            npu_graph_runner = get_or_create_runner(CUR_DEVICE)
            stream = npu_graph_runner.main_stream
            if npu_graph_runner.past_key_value is None:
                npu_graph_runner.past_key_value = past_key_value
            if npu_graph_runner.workspace is None:
                workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    q_nope,
                    kvCache,
                    kvCache,
                    query_rope=q_pe,
                    key_rope=kRopeCache,
                    num_heads=self.num_heads,
                    num_key_value_heads=1,
                    input_layout="BNSD",
                    atten_mask=None,
                    scale=self.softmax_scale,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    block_table=past_key_value.page_table_list[self.layer_idx],
                    block_size=past_key_value.page_size,
                    actual_seq_lengths_kv=past_key_value.position
                )
                npu_graph_runner.workspace = workspace
            attn_output = torch.zeros_like(q_nope, dtype=torch.float16, device=CUR_DEVICE)
            softmax_lse = torch.empty(1, dtype=torch.float16, device=CUR_DEVICE)
            npu_graph_runner.ifa_param.append((q_nope, kvCache, q_pe, kRopeCache, self.num_heads,
                                               self.softmax_scale, self.layer_idx, attn_output, softmax_lse))
            eventTmp = torch.npu.ExternalEvent()
            npu_graph_runner.event.append(eventTmp)
            eventTmp.wait(stream)
            eventTmp.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                kvCache,
                kvCache,
                workspace=npu_graph_runner.workspace,
                query_rope=q_pe,
                key_rope=kRopeCache,
                num_heads=self.num_heads,
                num_key_value_heads=1,
                input_layout="BNSD",
                atten_mask=None,
                scale=self.softmax_scale,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=past_key_value.page_table_list[self.layer_idx],
                block_size=past_key_value.page_size,
                actual_seq_lengths_kv=past_key_value.position,
                out=[attn_output, softmax_lse])
            handle = torch.npu.graph_task_group_end(stream)
            npu_graph_runner.handle.append(handle)
        else:
            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope,
                kvCache,
                kvCache,
                query_rope=q_pe,
                key_rope=kRopeCache,
                num_heads=self.num_heads,
                num_key_value_heads=1,
                input_layout="BNSD",
                atten_mask=None,
                scale=self.softmax_scale,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=past_key_value.page_table_list[self.layer_idx],
                block_size=past_key_value.page_size,
                actual_seq_lengths_kv=past_key_value.position,
            )

        attn_output = torch.einsum('b h q k, h k v -> b q h v', attn_output, self.out_absorb)
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.elewise_quant.execute([attn_output, self.o_proj.input_scale, self.o_proj.input_offset])[0]
        attn_output = self.matmulDequant_operation_aclnn.execute([attn_output, self.o_proj.weight,
                                                            self.o_proj.quant_bias, self.o_proj.deq_scale])[0]
        return attn_output, None, past_key_value

    def forward_windows(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        if q_len <= self.chunck_size:
            return self.forward_chunck(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                **kwargs
            )

        assert output_attentions is False, "output_attentions is not supported when using chunked attention"
        attn_output = None
        cur_idx = 0
        while cur_idx < q_len:
            if attention_mask is not None:
                chunk_mask = attention_mask[:, :, cur_idx:min(cur_idx + self.chunck_size, q_len), ...]
            else:
                # generate chunk_mask automatically.
                self.attn_mask = \
                    torch.zeros(1, 1, self.chunck_size, past_key_value.max_cache_len, device=hidden_states.device) \
                        if self.attn_mask is None \
                        else self.attn_mask
                self.attn_mask[:, :, :, cur_idx:min(cur_idx + self.chunck_size, past_key_value.max_cache_len)] = \
                    -65504.0 * torch.triu(torch.ones(self.chunck_size, self.chunck_size, device=hidden_states.device), diagonal=1) \
                        [:, :min(self.chunck_size, min(past_key_value.max_cache_len - cur_idx, self.chunck_size))]
                self.attn_mask[:, :, :, cur_idx + self.chunck_size:] = -65504.0
                self.attn_mask[:, :, :, :cur_idx] = 0
                chunk_mask = torch.narrow(self.attn_mask, 2, 0, min(self.chunck_size, q_len - cur_idx))

            cur_output, _, _ = self.forward_chunck(
                hidden_states[:, cur_idx:min(cur_idx + self.chunck_size, q_len), ...],
                chunk_mask,
                position_ids[:, cur_idx:min(cur_idx + self.chunck_size, q_len)],
                past_key_value,
                output_attentions,
                use_cache,
                cache_position[cur_idx:min(cur_idx + self.chunck_size, q_len)],
                **kwargs
            )
            cur_idx += self.chunck_size
            if attn_output is None:
                attn_output = cur_output
            else:
                attn_output = torch.cat((attn_output, cur_output), dim=-2)

        return attn_output, None, past_key_value

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        return self.forward_windows(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            **kwargs,
        )