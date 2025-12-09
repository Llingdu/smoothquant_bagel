import torch
import torch.nn as nn
import os
import json
import sys
import functools
from tqdm import tqdm
from accelerate import load_checkpoint_and_dispatch, infer_auto_device_map


# ================= 配置区域 =================
CODE_ROOT = "/root/autodl-tmp/smoothquant/Bagel-main"
WEIGHT_ROOT = "/root/autodl-tmp/Bagel-7B-MoT"
DATASET_PATH = "/root/autodl-tmp/data/wiki-text-2/wiki.train.tokens"
SCALE_SAVE_PATH = "act_scales/bagel-7b.pt"
CALIB_JSON = "calib_data.json"
NUM_SAMPLES = 64
SEQ_LEN = 512
# ===========================================

sys.path.append(CODE_ROOT)
try:
    from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2 import Qwen2Tokenizer
    from data.data_utils import add_special_tokens
    from modeling.bagel.qwen2_navit import NaiveCache
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    raise e

def prepare_data():
    if not os.path.exists(DATASET_PATH): raise FileNotFoundError(f"找不到: {DATASET_PATH}")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f: lines = f.readlines()
    count = 0
    with open(CALIB_JSON, 'w', encoding='utf-8') as f:
        for line in lines:
            text = line.strip()
            if len(text) > 50: 
                f.write(json.dumps({"text": text}) + "\n")
                count += 1
                if count >= NUM_SAMPLES: break
    print(f"1. [数据] 已准备 {count} 条样本")

def build_model_optimized():
    print("2. [模型] 开始组装 (MoT模式 + BF16)...")
    
    # --- Config 加载 ---
    llm_config = Qwen2Config.from_json_file(os.path.join(WEIGHT_ROOT, "llm_config.json"))
    llm_config.layer_module = "Qwen2MoTDecoderLayer" 
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    
    # 【修改点 1】强制使用 bfloat16
    llm_config.torch_dtype = torch.bfloat16 
    
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(WEIGHT_ROOT, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    
    # 【修改点 2】强制使用 bfloat16
    vit_config.torch_dtype = torch.bfloat16

    bagel_config = BagelConfig(visual_gen=False, visual_und=True, llm_config=llm_config, vit_config=vit_config, connector_act='gelu_pytorch_tanh')

    print("   正在初始化结构 (CPU)...")
    
    # 【修改点 3】初始化时就用 bfloat16
    with torch.device("cpu"):
        language_model = Qwen2ForCausalLM(llm_config).to(dtype=torch.bfloat16)
        vit_model = SiglipVisionModel(vit_config).to(dtype=torch.bfloat16)
        model = Bagel(language_model, vit_model, bagel_config).to(dtype=torch.bfloat16)
    
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    print("3. [加载] 使用手动 Device Map (强制分层)...")
    
    # 显存分配策略保持不变
    device_map = {}
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    
    gpu_layers = 10
    total_layers = 32
    
    print(f"   策略: 前 {gpu_layers} 层 -> GPU, 后 {total_layers - gpu_layers} 层 -> CPU")
    
    for i in range(total_layers):
        if i < gpu_layers:
            device_map[f"language_model.model.layers.{i}"] = 0
        else:
            device_map[f"language_model.model.layers.{i}"] = "cpu"
            
    device_map["language_model.model.norm"] = "cpu"
    device_map["language_model.model.norm_moe_gen"] = "cpu"
    device_map["language_model.lm_head"] = "cpu"
    
    device_map["vit_model"] = "cpu"
    device_map["vae2llm"] = "cpu"
    device_map["llm2vae"] = "cpu"
    device_map["time_embedder"] = "cpu"
    device_map["latent_pos_embed"] = "cpu"
    device_map["connector"] = "cpu"
    device_map["vit_pos_embed"] = "cpu"

    print("   开始加载权重...")
    torch.cuda.empty_cache()
    
    model = load_checkpoint_and_dispatch(
        model, 
        os.path.join(WEIGHT_ROOT, "ema.safetensors"), 
        device_map=device_map, 
        dtype=torch.bfloat16, 
        no_split_module_classes=["Qwen2MoTDecoderLayer", "Qwen2DecoderLayer"]
    )
    
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"✅ 加载完成！当前 GPU 显存占用: {gpu_mem:.2f} GB")
    
    return model

def get_act_scales(model, tokenizer, dataset_path, num_samples=128, seq_len=512):
    model.eval()
    act_scales = {}

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple): x = x[0]
        tensor = x.view(-1, x.shape[-1]).abs().detach().cpu().to(torch.float32)
        comming_max = torch.max(tensor, dim=0)[0]
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    hooks = []
    print("4. [Hook] 扫描并注册 Linear 层...")
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if "lm_head" in name: continue
            hooks.append(m.register_forward_hook(functools.partial(stat_input_hook, name=name)))
    print(f"   已注册 {len(hooks)} 个 Hook。")
    
    dataset = []
    with open(dataset_path, 'r') as f:
        for line in f: dataset.append(json.loads(line)["text"])

    print(f"5. [校准] 开始前向传播 (Samples={len(dataset)})...")
    
    llm_backbone = model.language_model.model
    device = "cuda:0" 
    num_hidden_layers = llm_backbone.config.num_hidden_layers
    embed_tokens = llm_backbone.embed_tokens
    for text in tqdm(dataset):
        inputs = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True)
        input_ids = inputs.input_ids.to(device)
        curr_seq_len = input_ids.shape[1]

        try:
            with torch.no_grad():
                
                past_key_values = NaiveCache(num_hidden_layers)
                packed_query_sequence = embed_tokens(input_ids).squeeze(0)
                query_lens = torch.tensor([curr_seq_len], dtype=torch.int32, device=device)
                packed_query_position_ids = torch.arange(curr_seq_len, dtype=torch.long, device=device)
                packed_query_indexes = torch.arange(curr_seq_len, dtype=torch.long, device=device)
                
                # 调用底层 forward_inference
                llm_backbone.forward_inference(
                    packed_query_sequence=packed_query_sequence,
                    query_lens=query_lens,
                    packed_query_position_ids=packed_query_position_ids,
                    packed_query_indexes=packed_query_indexes,
                    past_key_values=past_key_values,
                    
                    key_values_lens=None,
                    packed_key_value_indexes=None,
                    mode="gen"
                )
                
        except Exception as e:
            print(f"\n⚠️ 样本出错: {e}")
            import traceback
            traceback.print_exc()
            break 

    for h in hooks: h.remove()
    return act_scales

def main():
    prepare_data()
    try:
        model = build_model_optimized()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    tokenizer = Qwen2Tokenizer.from_pretrained(WEIGHT_ROOT)
    tokenizer, _, _ = add_special_tokens(tokenizer)

    act_scales = get_act_scales(model, tokenizer, CALIB_JSON, NUM_SAMPLES, SEQ_LEN)

    if act_scales:
        os.makedirs(os.path.dirname(SCALE_SAVE_PATH), exist_ok=True)
        torch.save(act_scales, SCALE_SAVE_PATH)
        print(f"\n✅ 校准完成！结果已保存至: {SCALE_SAVE_PATH}")
    else:
        print("\n❌ 失败：未收集到数据")

if __name__ == "__main__":
    main()
