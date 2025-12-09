import argparse
import base64
import itertools
import json
import os
import random
import sys
from io import BytesIO

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================
# 1. Bagel 代码根目录 (用于导入 eval.vlm.utils 和模型)
CODE_ROOT = "/root/autodl-tmp/smoothquant/Bagel-main"
# 2. 模型权重路径
MODEL_PATH = "/root/autodl-tmp/Bagel-7B-MoT"
# ===========================================

# 将 Bagel 代码加入系统路径，确保能 import eval.vlm.utils
sys.path.append(CODE_ROOT)

try:
    # 尝试从官方代码库导入工具
    from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation,apply_quantization
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print(f"请检查 CODE_ROOT 是否正确指向了包含 'eval' 文件夹的 Bagel-main 目录: {CODE_ROOT}")
    sys.exit(1)

# 修改数据集路径为你的绝对路径
ds_collections = {
    'mmbench_dev_en_20231003': {
        'root': '/root/autodl-tmp/data/mmbench/mmbench_dev_en_20231003.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'dev',
        'language': 'en'
    },
    'mmbench_test_en_20231003': {
        'root': '/root/autodl-tmp/data/mmbench/mmbench_test_en_20231003.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'test',
        'language': 'en'
    },
    # 也可以保留其他配置，以防万一需要
    'mmbench_dev_20230712': {
        'root': 'eval/vlm/data/mmbench/mmbench_dev_20230712.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'dev',
        'language': 'en'
    },
}

def collate_fn(batches):
    questions = [_['question'] for _ in batches]
    images = [_['images'] for _ in batches]
    conversation = [_['conversation'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    indexes = [_['index'] for _ in batches]
    options = [_['option'] for _ in batches]
    return questions, images, conversation, answers, indexes, options


class MMBenchDataset(torch.utils.data.Dataset):

    def __init__(self, root, prompt, language):
        self.df = pd.read_csv(root, sep='\t')
        self.prompt = prompt
        self.language = language

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image_str = self.df.iloc[idx]['image'] # 修改变量名以避免混淆
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else None

        image = Image.open(BytesIO(base64.b64decode(image_str))).convert('RGB')
        images = [image]

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }

        hint = self.load_from_df(idx, 'hint')
        if hint is not None:
            question = hint + '\n' + question
        for key, item in options.items():
            question += f'\n{key}. {item}'
        if self.language == 'cn':
            question = question + '\n' + self.prompt['cn']
        else:
            question = question + '\n' + self.prompt['en']

        images, conversation = process_conversation(images, question)

        return {
            'question': question,
            'images': images,
            'conversation': conversation,
            'answer': answer,
            'index': index,
            'option': options
        }

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def post_process(pred, option):
    pred = pred.strip()
    option_candidate = list(option.keys())
    if len(pred) == 1:
        return pred
    if len(pred) == 0:
        pred = "C"
    elif len(pred) != 1 and pred[0] in option_candidate:
        return pred[0]
    elif len(pred) != 1 and pred[0] not in option_candidate:
        for k, v in option.items():
            if v in pred:
                return k

    return pred


def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = MMBenchDataset(
            root=ds_collections[ds_name]['root'],
            prompt=prompt,
            language=ds_collections[ds_name]['language'],
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        outputs = []
        for _, (questions, images, conversation, answers, indexes, options) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # 直接调用官方提供的 model.chat
            pred = model.chat(
                tokenizer, 
                new_token_ids,
                image_transform,
                images=images[0], 
                prompt=conversation[0], 
                max_length=ds_collections[ds_name]['max_new_tokens'], 
            )
            preds = [post_process(pred, options[0])]

            for question, pred, answer, index in zip(questions, preds, answers, indexes):
                outputs.append({
                    'question': question,
                    'answer': pred,
                    'gt_answers': answer,
                    'index': int(index)
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            results_file = f'results_{ds_name}.xlsx'
            output_path = os.path.join(args.out_dir, results_file)
            df = pd.read_table(ds_collections[ds_name]['root'])
            cur_df = df.copy()
            if 'mmbench' in ds_name:
                # 根据实际数据列情况，可能需要调整 drop 的列
                cols_to_drop = ['hint', 'category', 'source', 'image', 'comment', 'l2-category']
                cur_df = cur_df.drop(columns=[c for c in cols_to_drop if c in cur_df.columns])
                if 'prediction' not in cur_df.columns:
                    cur_df['prediction'] = None
            else:
                cur_df = cur_df.drop(columns=['category', 'image'])
                if 'prediction' not in cur_df.columns:
                    cur_df['prediction'] = None
            
            for item in merged_outputs:
                cur_df.loc[df['index'] == item['index'], 'prediction'] = item['answer']

            cur_df.to_excel(output_path, index=False, engine='openpyxl')
            print('Results saved to {}'.format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 默认使用你提供的 dev 集
    parser.add_argument('--datasets', type=str, default='mmbench_dev_en_20231003')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='mmbench_results')
    parser.add_argument('--seed', type=int, default=0)
    # 修改默认模型路径
    parser.add_argument('--model-path', type=str, default=MODEL_PATH)
    parser.add_argument("--quant_mode", type=str, default="baseline", 
                        choices=["baseline", "naive", "smooth"], 
                        help="量化模式: baseline(不量化), naive(朴素W8A8), smooth(SmoothQuant)")
    parser.add_argument("--act_quant", type=str, default="per_tensor", 
                        choices=["per_token", "per_tensor"], 
                        help="激活量化粒度")
    parser.add_argument("--scale_path", type=str, default="act_scales/bagel-7b-mmbench.pt", 
                        help="校准文件路径")
    parser.add_argument("--bagel_mode", type=str, default="und", 
                        choices=["und", "gen"], 
                        help="Bagel模式: und (Understanding) 或 gen (Generation)")
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    # 分布式初始化
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    local_rank = int(os.getenv('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    # 使用官方提供的 loader
    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    model = apply_quantization(model, args)

    image_transform = build_transform()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')

    prompt = {
        'en': "Answer with the option's letter from the given choices directly.",
        'cn': '请直接回答选项字母。'
    }
    evaluate_chat_model()

#使用实例：
#baseline:torchrun --nproc_per_node=1 evaluate_mmbench.py --quant_mode baseline
#naive:torchrun --nproc_per_node=1 evaluate_mmbench.py --quant_mode naive
#smooth:torchrun --nproc_per_node=1 evaluate_mmbench.py --quant_mode smooth --scale_path act_scales/bagel-7b-mmbench.pt
