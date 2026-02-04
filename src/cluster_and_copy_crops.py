"""
DINOv2 + Tagger 特征：UMAP 降维后拼接，再 HDBSCAN 聚类，并将同一簇的图片复制到 cluster_X/。
输入为 metadata_crops.json 及对应特征 .pt。可设定使用 head 或 face 特征。
支持 --mode：both（默认）、dinov2、tagger，可仅用单一特征聚类。
- DINOv2：768 维 -> L2 归一化 -> UMAP 降到 dim_dino（默认 50）
- Tagger：取前 256 维 logits -> Sigmoid + L2 归一化 -> UMAP 降到 dim_tagger（默认 50）
- 拼接 [dino_umap, tagger_umap] 后聚类；复制时同时输出 crop、head、face 图到各自目录。
"""
import json
import random
import shutil
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

try:
    import umap
except ImportError:
    raise SystemExit("请安装: pip install umap-learn")

from sklearn.cluster import HDBSCAN

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DINOV2_FEATURES = SCRIPT_DIR / "metadata_crops_dinov2_features.pt"
DEFAULT_TAGGER_FEATURES = SCRIPT_DIR / "metadata_crops_tagger_features.pt"
DEFAULT_UUID_JSON = SCRIPT_DIR / "metadata_crops_grouped.json"
DEFAULT_CLUSTERED_JSON = SCRIPT_DIR / "metadata_crops_clustered.json"
OUTPUT_DIR = "clustered/characters"
OUTPUT_HEAD_DIR = "clustered/heads"
OUTPUT_FACE_DIR = "clustered/faces"
DIM_DINO = 64
DIM_TAGGER = 64
UMAP_NEIGHBORS = 40


def load_pt(path, weights_only=True):
    try:
        return torch.load(path, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(description="DINOv2 + Tagger UMAP 降维拼接 + HDBSCAN 聚类 + 按簇复制")
    parser.add_argument("--features", "--dinov2-features", type=Path, default=DEFAULT_DINOV2_FEATURES, help="DINOv2 特征 .pt")
    parser.add_argument("--tagger-features", type=Path, default=DEFAULT_TAGGER_FEATURES, help="Tagger 特征 .pt")
    parser.add_argument("--uuid-json", type=Path, default=DEFAULT_UUID_JSON, help="metadata_crops.json 路径")
    parser.add_argument("--mode", choices=["both", "dinov2", "tagger"], default="both", help="使用 both/dinov2/tagger 特征（默认 both）")
    parser.add_argument("--feature-type", choices=["head", "face"], default="face", help="使用 head 或 face 特征")
    parser.add_argument("--output-json", type=Path, default=None, help="聚类结果 JSON 路径（默认 metadata_crops_clustered_{feature-type}.json）")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="crop 图按簇复制的输出根目录")
    parser.add_argument("--output-head-dir", default=OUTPUT_HEAD_DIR, help="head 图按簇复制的输出根目录")
    parser.add_argument("--output-face-dir", default=OUTPUT_FACE_DIR, help="face 图按簇复制的输出根目录")
    parser.add_argument("--dim-dino", type=int, default=DIM_DINO, help="DINOv2 UMAP 目标维度")
    parser.add_argument("--dim-tagger", type=int, default=DIM_TAGGER, help="Tagger UMAP 目标维度")
    parser.add_argument("--umap-neighbors", type=int, default=UMAP_NEIGHBORS, help="UMAP n_neighbors")
    parser.add_argument("--min-cluster-size", type=int, default=3, help="HDBSCAN 最小簇大小")
    parser.add_argument("--min-samples", type=int, default=1, help="HDBSCAN min_samples")
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.50, help="HDBSCAN 簇合并距离阈值")
    parser.add_argument("--no-copy", action="store_true", help="仅聚类，不复制图片")
    parser.add_argument("--skip-noise", action="store_true", help="复制时跳过 cluster_id=-1")
    parser.add_argument("--limit", type=int, default=0, help="仅取前 N 条处理（0=全部）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，配合 --limit 使用")
    args = parser.parse_args()

    use_dino = args.mode in ("both", "dinov2")
    use_tagger = args.mode in ("both", "tagger")
    if use_dino and not args.features.is_file():
        raise SystemExit(f"DINOv2 特征不存在: {args.features}")
    if use_tagger and not args.tagger_features.is_file():
        raise SystemExit(f"Tagger 特征不存在: {args.tagger_features}")
    if not args.uuid_json.is_file():
        raise SystemExit(f"UUID JSON 不存在: {args.uuid_json}")

    if args.output_json is None:
        args.output_json = SCRIPT_DIR / f"metadata_crops_clustered_{args.feature_type}.json"

    base_dir = args.features.resolve().parent if use_dino else args.tagger_features.resolve().parent
    dinov2_dict = None
    tagger_dict = None
    feat_key = "face_features" if args.feature_type == "face" else "features"

    if use_dino:
        dinov2_raw = load_pt(args.features)
        dinov2_dict = dinov2_raw.get(feat_key, dinov2_raw) if isinstance(dinov2_raw, dict) else dinov2_raw
    if use_tagger:
        tagger_raw = load_pt(args.tagger_features)
        tagger_dict = tagger_raw.get(feat_key, tagger_raw) if isinstance(tagger_raw, dict) else tagger_raw

    with open(args.uuid_json, "r", encoding="utf-8") as f:
        uuid_data = json.load(f)

    # 取交集：至少需在所用特征中存在
    if use_dino and use_tagger:
        crop_paths = [k for k in uuid_data if k in dinov2_dict and k in tagger_dict]
    elif use_dino:
        crop_paths = [k for k in uuid_data if k in dinov2_dict]
    else:
        crop_paths = [k for k in uuid_data if k in tagger_dict]

    # 按 feature-type 筛选：仅用组内代表参与聚类
    if args.feature_type == "face":
        crop_paths = [k for k in crop_paths if uuid_data[k].get("first_in_face_group") is True]
        print(f"筛选 first_in_face_group=true: {len(crop_paths)} 条参与聚类")
    else:
        crop_paths = [k for k in crop_paths if uuid_data[k].get("first_in_head_group") is True]
        print(f"筛选 first_in_head_group=true: {len(crop_paths)} 条参与聚类")
    if not crop_paths:
        raise SystemExit("features 与 uuid 无交集，或筛选后无数据，请检查 crop_path 及 first_in_*_group 字段")
    if args.limit > 0:
        n = min(args.limit, len(crop_paths))
        if args.seed is not None:
            random.seed(args.seed)
        crop_paths = random.sample(crop_paths, n)
        print(f"随机抽取 {n} 条进行处理")

    N = len(crop_paths)
    parts = []

    if use_dino:
        X_dino = torch.stack([dinov2_dict[p] for p in crop_paths])
        X_dino = F.normalize(X_dino, p=2, dim=1).numpy()
        dim_dino = min(args.dim_dino, X_dino.shape[0] - 1, X_dino.shape[1])
        reducer_dino = umap.UMAP(
            n_components=dim_dino,
            n_neighbors=min(args.umap_neighbors, X_dino.shape[0] - 1),
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        emb_dino = reducer_dino.fit_transform(X_dino)
        parts.append(emb_dino)
        print(f"DINOv2 L2+UMAP: {X_dino.shape[1]} -> {dim_dino} 维")

    if use_tagger:
        feats_t = torch.stack([tagger_dict[p] for p in crop_paths], dim=0)[:, :256]
        probs = torch.sigmoid(feats_t)
        norm_t = F.normalize(probs, p=2, dim=1).numpy()
        dim_tagger = min(args.dim_tagger, norm_t.shape[0] - 1, norm_t.shape[1])
        reducer_tagger = umap.UMAP(
            n_components=dim_tagger,
            n_neighbors=min(args.umap_neighbors, norm_t.shape[0] - 1),
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        emb_tagger = reducer_tagger.fit_transform(norm_t)
        parts.append(emb_tagger)
        print(f"Tagger Sigmoid+L2+UMAP: {feats_t.shape[1]} -> {dim_tagger} 维")

    X = np.concatenate(parts, axis=1)
    print(f"拼接后: {X.shape[1]} 维，N={N}，范围 [{X.min():.3f}, {X.max():.3f}]，std={X.std():.3f}")

    # HDBSCAN
    print(f"HDBSCAN: min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}, cluster_selection_epsilon={args.cluster_selection_epsilon}")
    clusterer = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X)

    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    noise_ratio = n_noise / n_total if (n_total := len(labels)) else 0
    cluster_sizes = [(labels == cid).sum() for cid in set(labels) if cid != -1]
    avg_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
    print(f"聚类完成: {n_clusters} 个簇，{n_noise} 个噪声点（噪点占比 {noise_ratio:.1%}），平均簇大小 {avg_size:.2f}")
    top10 = sorted(((cid, int((labels == cid).sum())) for cid in set(labels) if cid != -1), key=lambda x: -x[1])[:10]
    print("最大的十个簇: " + ", ".join(f"cluster_{c}({s})" for c, s in top10))

    out_data = {}
    for path, label in zip(crop_paths, labels):
        rec = dict(uuid_data[path])
        rec["cluster_id"] = int(label)
        out_data[path] = rec

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"已保存到 {args.output_json}")

    if not args.no_copy:
        out_root = base_dir / args.output_dir
        head_root = base_dir / args.output_head_dir
        face_root = base_dir / args.output_face_dir
        if out_root.exists():
            shutil.rmtree(out_root)
        if head_root.exists():
            shutil.rmtree(head_root)
        if face_root.exists():
            shutil.rmtree(face_root)
        for crop_path, rec in out_data.items():
            cid = rec["cluster_id"]
            if args.skip_noise and cid == -1:
                continue
            subdir = out_root / f"cluster_{cid}"
            subdir.mkdir(parents=True, exist_ok=True)
            head_subdir = head_root / f"cluster_{cid}"
            head_subdir.mkdir(parents=True, exist_ok=True)
            face_subdir = face_root / f"cluster_{cid}"
            face_subdir.mkdir(parents=True, exist_ok=True)
            # 复制 crop 图（person_path 为实际 crop 文件路径）
            person_path = rec.get("person_path")
            if person_path:
                src_crop = (base_dir / person_path).resolve()
                if src_crop.is_file():
                    shutil.copy2(src_crop, subdir / Path(person_path).name)
            # 复制 head 图
            head_path = rec.get("head_path")
            if head_path:
                src_head = (base_dir / head_path).resolve()
                if src_head.is_file():
                    shutil.copy2(src_head, head_subdir / Path(head_path).name)
            # 复制 face 图
            face_path = rec.get("face_path")
            if face_path:
                src_face = (base_dir / face_path).resolve()
                if src_face.is_file():
                    shutil.copy2(src_face, face_subdir / Path(face_path).name)
        n_dirs = len([d for d in out_root.iterdir() if d.is_dir()])
        print(f"已复制 crop 到 {out_root}，head 到 {head_root}，face 到 {face_root}，共 {n_dirs} 个子目录")


if __name__ == "__main__":
    main()
