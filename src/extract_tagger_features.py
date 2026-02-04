"""
使用 Waifu Diffusion 1.4 Tagger (SmilingWolf/wd-v1-4-vit-tagger-v2) 对 metadata_crops.json
中的 head 和 face 图片提取特征。有 head_path 则提取 head 特征，有 face_path 则提取 face 特征。
特征字典 key 为 uuid，输出 features（head）与 face_features（face）。
该模型仅提供 ONNX 格式，无 transformers processor，故使用 ONNX Runtime 推理。
特征过滤：按 selected_tags.csv 取 top-N 标签（CSV 已按 count 排序），可选排除黑名单（表情/构图/数量类）。
输出原始 logits 特征（不做 L2 归一化）。
预处理逻辑参考：https://huggingface.co/spaces/SmilingWolf/wd-tagger
"""
import csv
import os
from typing import List, Optional, Tuple
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort
from huggingface_hub import hf_hub_download


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = SCRIPT_DIR / "metadata_crops.json"
DEFAULT_FEATURES = SCRIPT_DIR / "metadata_crops_tagger_features.pt"
DEFAULT_MODEL = "SmilingWolf/wd-v1-4-vit-tagger-v2"
DEFAULT_TOP_N = 512  # 保留的通用标签数

# 黑名单关键词：标签名包含任一关键词则排除（聚类头像时排除表情/构图/数量类）
EXCLUDE_KEYWORDS = [
    "blush", "smile", "grin", "open_mouth", "closed_eyes", "wink", "pout",
    "shouting", "sigh", "tongue", "tears", "expressionless", "narrowed_eyes",
    "half-closed_eyes", "serious", "surprised", "parted_lips", "bored",
    "looking_at_viewer", "looking_away", "facing_viewer", "profile",
    "portrait", "close-up", "upper_body", "lower_body",
    "1girl", "solo", "multiple_girls", "highres", "absurdres",
    "watermark", "text", "blurry", "depth_of_field",
]


def load_selected_tag_indices(
    model_repo: str,
    top_n: int = 512,
    exclude_keywords: Optional[List[str]] = None,
) -> Tuple[List[int], List[str]]:
    """
    从 selected_tags.csv 加载 top-N 标签索引。CSV 已按 count 排序。
    exclude_keywords：先取 top-N，再排除黑名单，输出维度会减少。
    None 表示不排除，保持 top-N 维。
    返回 (indices, tag_names)，indices 对应 logits 输出维度顺序。
    """
    csv_path = hf_hub_download(model_repo, "selected_tags.csv")
    indices = []
    names = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            indices.append(i)
            names.append(row.get("name", ""))
            if len(indices) >= top_n:
                break
    # 排除黑名单：移除后维度减少
    if exclude_keywords:
        keep_indices = []
        keep_names = []
        for idx, name in zip(indices, names):
            tag_lower = (name or "").lower()
            if any(kw in tag_lower for kw in exclude_keywords):
                continue
            keep_indices.append(idx)
            keep_names.append(name)
        indices, names = keep_indices, keep_names
    return indices, names


def prepare_image(image: Image.Image, target_size: int) -> np.ndarray:
    """按 WD14 Tagger 要求预处理：RGB、pad 成正方形、resize、BGR、float32"""
    image = image.convert("RGBA")  # 确保 RGBA，避免 alpha_composite 报 "images do not match"
    canvas = Image.new("RGBA", image.size, (255, 255, 255))
    canvas.alpha_composite(image)
    image = canvas.convert("RGB")
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(image, (pad_left, pad_top))
    if max_dim != target_size:
        resample = getattr(Image, "Resampling", Image).BICUBIC
        padded = padded.resize((target_size, target_size), resample)
    arr = np.asarray(padded, dtype=np.float32)
    arr = arr[:, :, ::-1]  # RGB -> BGR
    return arr


def main():
    parser = argparse.ArgumentParser(description="Waifu Diffusion Tagger 提取头部图标签特征（按 head_path 读图，ONNX）")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="含 head_path 的 JSON 路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_FEATURES, help="特征输出路径（.pt）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Tagger 模型 repo")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="保留的 top-N 标签数（CSV 按 count 排序）")
    parser.add_argument("--no-exclude", action="store_true", help="不排除黑名单标签（表情/构图/数量类）")
    parser.add_argument("--limit", type=int, default=0, help="随机抽取 N 条处理（0=全部）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，便于复现")
    args = parser.parse_args()

    base_dir = args.json.resolve().parent
    if not args.json.is_file():
        raise SystemExit(f"JSON 不存在: {args.json}")

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 收集需处理的 uuid：有 head_path 或 face_path 的条目
    uuids = list(data.keys())
    if args.limit > 0:
        if args.seed is not None:
            random.seed(args.seed)
        n = min(args.limit, len(uuids))
        uuids = random.sample(uuids, n)
        print(f"随机抽取 {n} 条进行处理")

    # 预选 top-N 标签索引（CSV 按 count 排序），默认排除黑名单
    exclude_kw = None if args.no_exclude else EXCLUDE_KEYWORDS
    tag_indices, tag_names = load_selected_tag_indices(
        args.model, top_n=args.top_n, exclude_keywords=exclude_kw
    )
    n_removed = args.top_n - len(tag_indices) if exclude_kw else 0
    print(f"预选 {len(tag_indices)} 标签" + (f"（排除黑名单 {n_removed} 个）" if exclude_kw else ""))

    model_path = hf_hub_download(args.model, "model.onnx")
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    inp_shape = sess.get_inputs()[0].shape
    target_size = int(inp_shape[1]) if len(inp_shape) >= 2 else 448

    features_dict = {}  # uuid -> head 特征
    face_features_dict = {}  # uuid -> face 特征

    def _extract_one(img_path, uuid_key, out_dict, label):
        if not img_path.is_file():
            tqdm.write(f"跳过（{label} 文件不存在）: {img_path}")
            return
        try:
            img = Image.open(img_path)
            img.load()
            arr = prepare_image(img, target_size)
        except Exception as e:
            tqdm.write(f"跳过（{label} 读取失败）: {e}")
            return
        one = np.expand_dims(arr, axis=0)
        out = sess.run([output_name], {input_name: one})[0]
        full_logits = out[0]
        selected = full_logits[tag_indices]
        vec = torch.tensor(selected, dtype=torch.float32)
        out_dict[uuid_key] = vec

    for p in tqdm(uuids, desc="提取 Tagger 特征", unit="条"):
        rec = data[p]
        head_path = rec.get("head_path")
        face_path = rec.get("face_path")
        if head_path:
            _extract_one((base_dir / head_path).resolve(), p, features_dict, "head")
        if face_path:
            _extract_one((base_dir / face_path).resolve(), p, face_features_dict, "face")

    out_data = {
        "features": features_dict,
        "face_features": face_features_dict,
        "tag_indices": tag_indices,
        "tag_names": tag_names,
    }
    torch.save(out_data, args.output)
    n_head = len(features_dict)
    n_face = len(face_features_dict)
    dim = list(features_dict.values())[0].shape[-1] if features_dict else (list(face_features_dict.values())[0].shape[-1] if face_features_dict else 0)
    print(f"完成: head {n_head} 条，face {n_face} 条，维度 {dim}")
    print(f"已保存到 {args.output}")


if __name__ == "__main__":
    main()
