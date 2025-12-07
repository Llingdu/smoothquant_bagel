import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralRMSNorm,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralRMSNorm,
)
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer


from modeling.bagel.qwen2_navit import (
    Qwen2DecoderLayer, 
    Qwen2MoEDecoderLayer, 
    Qwen2MoTDecoderLayer,
    Qwen2RMSNorm
)

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, (Qwen2RMSNorm,LlamaRMSNorm, MistralRMSNorm, MixtralRMSNorm))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + ".fc1"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, FalconDecoderLayer):
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            fc1 = module.mlp.dense_h_to_4h

            if (
                not module.config.new_decoder_architecture
                and module.config.parallel_attn
            ):
                attn_ln = module.input_layernorm
                smooth_ln_fcs(attn_ln, [qkv, fc1], qkv_input_scales, alpha)
            else:
                attn_ln = (
                    module.ln_attn
                    if module.config.new_decoder_architecture
                    else module.input_layernorm
                )
                ffn_ln = (
                    module.ln_mlp
                    if module.config.new_decoder_architecture
                    else module.post_attention_layernorm
                )
                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
                smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer)):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
        elif isinstance(module, MixtralDecoderLayer):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.block_sparse_moe.gate]
            for expert in module.block_sparse_moe.experts:
                fcs.append(expert.w1)
                fcs.append(expert.w3)
            fcs_input_scales = scales[name + ".block_sparse_moe.gate"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)



@torch.no_grad()
def smooth_bagel(model, scales, alpha=0.5, mode="gen"):
    """
    Bagel 专用平滑函数。
    mode: 'und' 或 'gen'。
    如果 mode='gen' 但 scales 里缺少生成层的统计数据，会自动复用文本层的数据(Proxy Scale)。
    """
    print(f"Applying Bagel Smoothing (Mode: {mode})...")
    
    for name, module in model.named_modules():
        if isinstance(module, (Qwen2DecoderLayer, Qwen2MoEDecoderLayer, Qwen2MoTDecoderLayer)):
            
            # 定义前缀，Bagel 的层名通常包含 language_model.model.layers.X
            # scales 字典里的 key 必须和这里匹配
            layer_prefix = name 
            
            # Attention
            
            attn_ln = module.input_layernorm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            
            qkv_scale_key = layer_prefix + ".self_attn.q_proj"
            if qkv_scale_key in scales:
                smooth_ln_fcs_llama_like(attn_ln, qkv, scales[qkv_scale_key], alpha)
            else:
                print(f"[Warning] Missing scale for {qkv_scale_key}")

            #gen
            if mode == "gen" and hasattr(module, 'input_layernorm_moe_gen'):
                attn_ln_gen = module.input_layernorm_moe_gen
                qkv_gen = [
                    module.self_attn.q_proj_moe_gen,
                    module.self_attn.k_proj_moe_gen,
                    module.self_attn.v_proj_moe_gen,
                ]
                
                gen_scale_key = layer_prefix + ".self_attn.q_proj_moe_gen"
                
                if gen_scale_key in scales:
                    # Case 1: 有真实的生成数据校准 scale
                    smooth_ln_fcs_llama_like(attn_ln_gen, qkv_gen, scales[gen_scale_key], alpha)
                elif qkv_scale_key in scales:
                    # Case 2: 没有生成数据，借用文本 scale (Proxy Strategy)
                    smooth_ln_fcs_llama_like(attn_ln_gen, qkv_gen, scales[qkv_scale_key], alpha)

            #MLP
            ffn_ln = module.post_attention_layernorm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            
            mlp_scale_key = layer_prefix + ".mlp.gate_proj"
            if mlp_scale_key in scales:
                smooth_ln_fcs_llama_like(ffn_ln, fcs, scales[mlp_scale_key], alpha)

            if mode == "gen" and hasattr(module, 'mlp_moe_gen'):
                # 确定 Norm 层
                if hasattr(module, 'post_attention_layernorm_moe_gen'):
                    # MoT: 独立的 Norm
                    ffn_ln_gen = module.post_attention_layernorm_moe_gen
                else:
                    # MoE: 共享的 Norm (通常不建议混合平滑，但如果必须做...)
                    ffn_ln_gen = module.post_attention_layernorm
                
                fcs_gen = [module.mlp_moe_gen.gate_proj, module.mlp_moe_gen.up_proj]
                
                gen_mlp_key = layer_prefix + ".mlp_moe_gen.gate_proj"
                
                if gen_mlp_key in scales:
                    smooth_ln_fcs_llama_like(ffn_ln_gen, fcs_gen, scales[gen_mlp_key], alpha)
                elif mlp_scale_key in scales:
                    smooth_ln_fcs_llama_like(ffn_ln_gen, fcs_gen, scales[mlp_scale_key], alpha)
