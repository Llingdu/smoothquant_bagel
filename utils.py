# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

import os
import yaml
from accelerate import load_checkpoint_and_dispatch
import sys
import torch

from data.data_utils import add_special_tokens, pil_img2rgb
from modeling.bagel import (
    BagelConfig, 
    Bagel, 
    Qwen2Config, 
    Qwen2ForCausalLM, 
    SiglipVisionConfig, 
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from safetensors.torch import load_file

from data.transforms import ImageTransform

MODEL_PATH = "/root/autodl-tmp/Bagel-7B-MoT"
# 重写加载模型，vae部分放到cpu上
def load_model_and_tokenizer(args):
    print("🚀 [Optimized Loader] 开始加载 Bagel 模型 (BF16)...")
    
    # 1. Config 加载
    llm_config = Qwen2Config.from_json_file(os.path.join(MODEL_PATH, "llm_config.json"))
    llm_config.layer_module = "Qwen2MoTDecoderLayer" 
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.torch_dtype = torch.bfloat16 
    
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(MODEL_PATH, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    vit_config.torch_dtype = torch.bfloat16

    bagel_config = BagelConfig(visual_gen=False, visual_und=True, llm_config=llm_config, vit_config=vit_config, connector_act='gelu_pytorch_tanh')

    # 2. 初始化空结构 (CPU)
    print("   正在初始化模型骨架...")
    with torch.device("cpu"):
        language_model = Qwen2ForCausalLM(llm_config).to(dtype=torch.bfloat16)
        vit_model = SiglipVisionModel(vit_config).to(dtype=torch.bfloat16)
        model = Bagel(language_model, vit_model, bagel_config).to(dtype=torch.bfloat16)
    
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    # 3. 显存分配策略 (关键！)
    # MMBench 需要视觉模块在 GPU 上
    print("   应用显存分配策略 (Vision -> GPU)...")
    device_map = {}
    
    # 核心计算模块 -> GPU 0
    device_map["language_model"] = 0 
    device_map["vit_model"] = 0
    device_map["connector"] = 0
    
    # 纯生成模块 (MMBench 用不到) -> CPU 以省显存
    device_map["vae2llm"] = "cpu"
    device_map["llm2vae"] = "cpu"
    device_map["time_embedder"] = "cpu"
    device_map["latent_pos_embed"] = "cpu"
    device_map["vit_pos_embed"] = "cpu"

    # 4. 加载权重
    print("   正在加载权重 (ema.safetensors)...")
    model = load_checkpoint_and_dispatch(
        model, 
        os.path.join(MODEL_PATH, "ema.safetensors"), 
        device_map=device_map, 
        dtype=torch.bfloat16, 
        no_split_module_classes=["Qwen2MoTDecoderLayer"]
    )
    
    # 5. Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    
    mem_used = torch.cuda.memory_allocated() / 1024**3
    print(f"✅ 模型加载成功! 当前显存占用: {mem_used:.2f} GB")
    
    return model, tokenizer, new_token_ids


def build_transform():
    with open("/root/autodl-tmp/smoothquant/Bagel-main/data/configs/example.yaml", "r") as f:
        data_config = yaml.safe_load(f)

    max_image_size = data_config['vlm_sft']['image_transform_args']['max_image_size']
    min_image_size = data_config['vlm_sft']['image_transform_args']['min_image_size']
    image_stride = data_config['vlm_sft']['image_transform_args']['image_stride']
    max_pixels = data_config['vlm_sft']['image_transform_args']['max_pixels']

    image_transform = ImageTransform(
        max_image_size=max_image_size,
        min_image_size=min_image_size,
        image_stride=image_stride,
        max_pixels=max_pixels,
    )

    return image_transform


def process_conversation(images, conversation):
    images = [pil_img2rgb(image) for image in images]
    return images, conversation

sys.path.append("/root/autodl-tmp/smoothquant")
from smooth import smooth_bagel
from fake_quant import quantize_bagel

#量化模式选择
def apply_quantization(model, args):
    """
    根据 args 参数决定执行哪种量化策略。
    
    args 需要包含以下属性:
      - quant_mode: 'baseline', 'naive', 'smooth'
      - act_quant: 'per_token', 'per_tensor' (默认 per_token)
      - scale_path: 校准文件路径 (仅 smooth 模式需要)
      - bagel_mode: 'und' (默认) 或 'gen'
    """
    
    # 1. Baseline: 不做任何处理，直接返回
    if args.quant_mode == 'baseline':
        print(">> Quantization: None (Baseline)")
        return model

    # 2. Naive Quantization: 直接 W8A8，不平滑
    elif args.quant_mode == 'naive':
        print(">> Quantization: Naive W8A8")
        model = quantize_bagel(
            model, 
            weight_quant="per_channel", 
            act_quant=args.act_quant, 
            quantize_bmm_input=True, 
            mode=args.bagel_mode
        )
        return model

    # 3. SmoothQuant: 平滑 + W8A8
    elif args.quant_mode == 'smooth':
        print(">> Quantization: SmoothQuant W8A8")
        
        if not os.path.exists(args.scale_path):
            raise FileNotFoundError(f"校准文件不存在: {args.scale_path}")
            
        print(f"   Loading scales from: {args.scale_path}")
        act_scales = torch.load(args.scale_path)
        
        # 执行平滑
        smooth_bagel(model, act_scales, alpha=0.7, mode=args.bagel_mode)
        
        # 执行量化
        model = quantize_bagel(
            model, 
            weight_quant="per_channel", 
            act_quant=args.act_quant, 
            quantize_bmm_input=True, 
            mode=args.bagel_mode
        )
        return model
    
    else:
        raise ValueError(f"Unknown quant_mode: {args.quant_mode}")
