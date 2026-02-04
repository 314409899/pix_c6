"""
使用 DINOv2 对 metadata_crops.json 中的 head 和 face 图片提取视觉特征。
有 head_path 则提取 head 特征，有 face_path 则提取 face 特征。
输出 features（head）与 face_features（face），key 为 uuid。
特征为 CLS token 嵌入（facebook/dinov2-base 为 768 维），保存到 .pt 文件。
"""
import os
# 必须在 import huggingface_hub 或 datasets 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import random
import argparse
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = SCRIPT_DIR / "metadata_crops.json"
DEFAULT_FEATURES = SCRIPT_DIR / "metadata_crops_dinov2_features.pt"
BATCH_SIZE = 8


def main():
    parser = argparse.ArgumentParser(description="DINOv2 提取头部图视觉特征（按 head_path 读图）")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="含 head_path 的 JSON 路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_FEATURES, help="特征输出路径（.pt）")
    parser.add_argument("--model", default="facebook/dinov2-base", help="DINOv2 模型，如 facebook/dinov2-base / dinov2-small")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="批量大小")
    parser.add_argument("--device", default=None, help="设备，如 cuda:0 或 cpu")
    parser.add_argument("--limit", type=int, default=0, help="随机抽取 N 条处理（0=全部）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，便于复现")
    args = parser.parse_args()

    try:
        from transformers import AutoImageProcessor, AutoModel
        from PIL import Image
        from tqdm import tqdm
    except ImportError:
        raise SystemExit("请安装: pip install transformers torch pillow tqdm")

    base_dir = args.json.resolve().parent
    if not args.json.is_file():
        raise SystemExit(f"JSON 不存在: {args.json}")

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 收集 head 与 face 的 (uuid, img_path) 列表
    uuids = list(data.keys())
    if args.limit > 0:
        if args.seed is not None:
            random.seed(args.seed)
        n = min(args.limit, len(uuids))
        uuids = random.sample(uuids, n)
        print(f"随机抽取 {n} 条进行处理")

    head_items = [(p, (base_dir / data[p]["head_path"]).resolve()) for p in uuids if data[p].get("head_path")]
    face_items = [(p, (base_dir / data[p]["face_path"]).resolve()) for p in uuids if data[p].get("face_path")]

    if not head_items and not face_items:
        raise SystemExit("JSON 中无 head_path 或 face_path，请检查数据")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    features_dict = {}
    face_features_dict = {}

    def _process_batch(items, out_dict, label):
        for i in tqdm(range(0, len(items), args.batch_size), desc=f"提取 {label} 特征", unit="batch"):
            batch = items[i : i + args.batch_size]
            images = []
            valid_uuids = []
            for p, img_path in batch:
                if not img_path.is_file():
                    tqdm.write(f"跳过（{label} 文件不存在）: {img_path}")
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    valid_uuids.append(p)
                except Exception as e:
                    tqdm.write(f"跳过（{label} 读取失败）: {e}")
                    continue
            if not images:
                continue
            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            cls_embeds = outputs.last_hidden_state[:, 0, :].cpu()
            for uuid, feat in zip(valid_uuids, cls_embeds):
                out_dict[uuid] = feat

    with torch.no_grad():
        _process_batch(head_items, features_dict, "head")
        _process_batch(face_items, face_features_dict, "face")

    out_data = {
        "features": features_dict,
        "face_features": face_features_dict,
    }
    torch.save(out_data, args.output)
    n_head = len(features_dict)
    n_face = len(face_features_dict)
    dim = list(features_dict.values())[0].shape[-1] if features_dict else (list(face_features_dict.values())[0].shape[-1] if face_features_dict else 0)
    print(f"完成: head {n_head} 条，face {n_face} 条，维度 {dim}")
    print(f"已保存到 {args.output}")


if __name__ == "__main__":
    main()
