import torch
import torch.nn as nn
import os
import json
import sys
import tqdm
import argparse
from accelerate import load_checkpoint_and_dispatch, infer_auto_device_map

# ================= 配置区域 =================
# 请根据实际情况修改路径
CODE_ROOT = "/root/autodl-tmp/smoothquant/Bagel-main"
WEIGHT_ROOT = "/root/autodl-tmp/Bagel-7B-MoT"
TEST_DATA_PATH = "/root/autodl-tmp/data/wiki-text-2/wiki.test.tokens" # 注意这里是 test 集
ACT_SCALES_PATH = "/root/autodl-tmp/smoothq_bagel/bagel-7b.pt"
# ===========================================

# 添加路径以导入 Bagel 代码
sys.path.append(CODE_ROOT)
try:
    from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2 import Qwen2Tokenizer
    from data.data_utils import add_special_tokens
    from modeling.bagel.qwen2_navit import NaiveCache
except ImportError as e:
    print(f"❌ Bagel 代码导入失败，请检查 CODE_ROOT: {e}")
    sys.exit(1)

# 导入 SmoothQuant (假设你已经安装或在当前目录)
from smooth import smooth_bagel
from fake_quant import quantize_bagel

def build_model_optimized():
    """
    显存优化版的模型加载函数 (BF16 + CPU Offload)
    """
    print("1. [模型] 开始组装 (MoT模式 + BF16 + CPU Offload)...")
    
    # 1. Config 加载
    llm_config = Qwen2Config.from_json_file(os.path.join(WEIGHT_ROOT, "llm_config.json"))
    llm_config.layer_module = "Qwen2MoTDecoderLayer" 
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.torch_dtype = torch.bfloat16 # 强制 BF16
    
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(WEIGHT_ROOT, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    vit_config.torch_dtype = torch.bfloat16

    bagel_config = BagelConfig(visual_gen=False, visual_und=True, llm_config=llm_config, vit_config=vit_config, connector_act='gelu_pytorch_tanh')

    # 2. 初始化空结构 (CPU)
    print("   正在初始化模型结构...")
    with torch.device("cpu"):
        language_model = Qwen2ForCausalLM(llm_config).to(dtype=torch.bfloat16)
        vit_model = SiglipVisionModel(vit_config).to(dtype=torch.bfloat16)
        model = Bagel(language_model, vit_model, bagel_config).to(dtype=torch.bfloat16)
    
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    # 3. 定义 Device Map (显存分配策略)
    # 这里的策略：LLM 的前 10 层进 GPU，剩下的和多模态组件全部扔 CPU
    # 这样可以保证显存占用控制在 15GB 左右，留出空间给 PPL 计算
    print("   应用显存分配策略...")
    device_map = {}
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    
    gpu_layers = 10 
    total_layers = 32
    
    for i in range(total_layers):
        if i < gpu_layers:
            device_map[f"language_model.model.layers.{i}"] = 0
        else:
            device_map[f"language_model.model.layers.{i}"] = 0
            
    # 将其他非核心计算部分全部移到 CPU
    device_map["language_model.model.norm"] = "cpu"
    device_map["language_model.model.norm_moe_gen"] = "cpu"
    device_map["language_model.lm_head"] = 0
    
    device_map["vit_model"] = "cpu"
    device_map["vae2llm"] = "cpu"
    device_map["llm2vae"] = "cpu"
    device_map["time_embedder"] = "cpu"
    device_map["latent_pos_embed"] = "cpu"
    device_map["connector"] = "cpu"
    device_map["vit_pos_embed"] = "cpu"

    # 4. 加载权重
    print("   开始加载权重 (ema.safetensors)...")
    torch.cuda.empty_cache()
    
    model = load_checkpoint_and_dispatch(
        model, 
        os.path.join(WEIGHT_ROOT, "ema.safetensors"), 
        device_map=device_map, 
        dtype=torch.bfloat16, 
        no_split_module_classes=["Qwen2MoTDecoderLayer", "Qwen2DecoderLayer"]
    )
    
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"✅ 模型加载完成！当前 GPU 显存占用: {gpu_mem:.2f} GB")
    
    return model

class Evaluator:
    def __init__(self, dataset_path, tokenizer, device, seq_len=2048):
        self.tokenizer = tokenizer
        self.device = device
        self.seq_len = seq_len
        
        print(f"   读取测试集: {dataset_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"找不到测试集: {dataset_path}")
            
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        print("   正在编码测试集...")
        self.encodings = tokenizer(text, return_tensors="pt")

    @torch.no_grad()
    def evaluate(self, model):
        # 获取 Qwen2ForCausalLM
        lm_model = model.language_model
        # 获取内部的 Qwen2Model (Backbone)
        backbone = lm_model.model
        # 获取 Embedding 层
        embed_tokens = backbone.embed_tokens
        
        lm_model.eval()
        
        max_length = self.seq_len
        stride = 512
        seq_len = self.encodings.input_ids.size(1)
        num_layers = backbone.config.num_hidden_layers

        nlls = []
        prev_end_loc = 0
        
        print(f"   开始计算 PPL (Total Tokens: {seq_len})...")
        pbar = tqdm.tqdm(range(0, seq_len, stride))
        
        loss_fct = nn.CrossEntropyLoss()

        for begin_loc in pbar:
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            
            # 1. 准备 Input IDs
            input_ids = self.encodings.input_ids[:, begin_loc:end_loc].to("cuda:0")
            curr_seq_len = input_ids.shape[1]
            
            # 2. 准备 Labels (用于计算 Loss)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100 

            with torch.no_grad():
                # --- 手动构造 Bagel 需要的 NaViT 参数 ---
                
                # A. 获取 Embedding (Batch=1, 所以 squeeze)
                packed_query_sequence = embed_tokens(input_ids).squeeze(0)
                
                # B. 构造 Lens 和 Position IDs
                query_lens = torch.tensor([curr_seq_len], dtype=torch.int32, device="cuda:0")
                packed_query_position_ids = torch.arange(curr_seq_len, dtype=torch.long, device="cuda:0")
                packed_query_indexes = torch.arange(curr_seq_len, dtype=torch.long, device="cuda:0")
                
                # C. 构造空的 Cache (必须传，否则报错)
                past_key_values = NaiveCache(num_layers)

                # 3. 前向传播 (获取 Hidden States)
                # 注意：这里我们使用 mode="und" (理解模式)，因为 PPL 是纯文本任务
                # 如果你想测试生成通路的量化效果，可以改为 mode="gen"
                outputs = backbone.forward_inference(
                    packed_query_sequence=packed_query_sequence,
                    query_lens=query_lens,
                    packed_query_position_ids=packed_query_position_ids,
                    packed_query_indexes=packed_query_indexes,
                    past_key_values=past_key_values,
                    key_values_lens=None,
                    packed_key_value_indexes=None,
                    mode="und", # 建议纯文本 PPL 用 und，如果必须测试 gen 通路请改为 "gen"
                    packed_vae_token_indexes=None, # gen 模式下可能需要这些参数，und 不需要
                    packed_text_indexes=None
                )
                
                # outputs 是 BaseNavitOutputWithPast，取出 hidden_states
                hidden_states = outputs.packed_query_sequence
                
                # 4. 计算 Logits (手动过 lm_head)
                # 注意：lm_head 可能在 CPU 上 (如果使用了 device_map)
                # 我们需要确保 hidden_states 和 lm_head 在同一设备
                lm_head_device = lm_model.lm_head.weight.device
                hidden_states = hidden_states.to(lm_head_device)
                
                logits = lm_model.lm_head(hidden_states)
                
                # 5. 手动计算 Loss (Shift Logits)
                # 语言模型的标准操作：用 t 时刻预测 t+1 时刻
                # Logits: [0, 1, ... N-1] -> Labels: [1, 2, ... N]
                
                # 调整维度以适配 CrossEntropyLoss
                # Logits: (Seq_Len, Vocab) -> (Seq_Len-1, Vocab)
                shift_logits = logits[:-1, :].contiguous()
                # Labels: (1, Seq_Len) -> (Seq_Len-1)
                shift_labels = target_ids.squeeze(0)[1:].to(lm_head_device).contiguous()
                
                loss = loss_fct(shift_logits, shift_labels)
                neg_log_likelihood = loss
            
            nlls.append(neg_log_likelihood)
            
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        # 汇总 PPL
        ppl = torch.exp(torch.stack(nlls).to("cpu").mean()) # 移回 CPU 计算平均
        return ppl.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "naive", "smooth"], help="实验模式")
    parser.add_argument("--act_quant", type=str, default="per_token", choices=["per_token", "per_tensor"], help="激活量化方式")
    args = parser.parse_args()

    # 1. 加载模型
    model = build_model_optimized()
    
    # 2. 加载 Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(WEIGHT_ROOT)
    tokenizer, _, _ = add_special_tokens(tokenizer)

    # 3. 根据模式处理模型
    print(f"\n[实验模式] {args.mode.upper()}")
    
    if args.mode == "baseline":
        print("   保持 FP16/BF16 基线...")
        
    elif args.mode == "naive":
        print(f"   应用朴素量化 (Naive W8A8) - {args.act_quant} ...")
        model = quantize_bagel(
            model, 
            weight_quant="per_channel", 
            act_quant=args.act_quant, 
            quantize_bmm_input=True,
            mode="und" 
        )
        
    elif args.mode == "smooth":
        print(f"   应用 SmoothQuant (Smooth + W8A8) - {args.act_quant} ...")
        
        if not os.path.exists(ACT_SCALES_PATH):
            print(f"❌ 错误：找不到校准数据 {ACT_SCALES_PATH}")
            return

        print("   正在加载校准数据...")
        act_scales = torch.load(ACT_SCALES_PATH)
        
        print("   执行平滑操作...")
        smooth_bagel(model, act_scales, alpha=0.5, mode="und")
        
        print("   执行量化操作...")
        model = quantize_bagel(
            model, 
            weight_quant="per_channel", 
            act_quant=args.act_quant, 
            quantize_bmm_input=True,
            mode="und"
        )

    # 4. 执行评估
    print("\n[开始评估]...")
    evaluator = Evaluator(TEST_DATA_PATH, tokenizer, "cuda")
    ppl = evaluator.evaluate(model)
    
    print("\n" + "="*40)
    print(f" Model: Bagel-7B-MoT")
    print(f" Mode:  {args.mode}")
    print(f" Quant: {args.act_quant if args.mode != 'baseline' else 'None'}")
    print(f" PPL:   {ppl:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
