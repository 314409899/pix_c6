"""
对 metadata.json 中按顺序排列的图片做连续帧分组：相邻且相似度极高的帧归为同一组。
每条记录增加 group_id、similarity_to_prev、quality_score、is_best_in_group 等字段。
支持多种相似度算法（imagehash）；分组后计算每张图清晰度/质量，选出组内最佳。
"""
import json
import shutil
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageOps
from tqdm import tqdm
import imagehash

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA = SCRIPT_DIR / "metadata.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "metadata_grouped.json"
VIS_DIR = "group"
RESIZE_SIZE = (64, 64)
DEFAULT_THRESHOLD = 0.7  # 相似度 >= 此值视为同一组（phash: 1 - 汉明距离/64；corr: 相关系数）
DEFAULT_WORKERS = 32
PHASH_SIZE = 8  # 8x8=64 位


def _load_thumb_corr(args):
    """加载单张图为 64x64 灰度向量，供 corr 方法。"""
    path, size = args
    if not path.is_file():
        return None
    try:
        resample = getattr(Image, "Resampling", Image).LANCZOS
        img = Image.open(path).convert("L").resize(size, resample)
        return np.asarray(img, dtype=np.float32).flatten()
    except Exception:
        return None


def _normalize_brightness(img: Image.Image, mode: str) -> Image.Image:
    """亮度/对比度归一化，减轻帧间整体变白/变暗的影响。"""
    if not mode or mode == "none":
        return img
    gray = img.convert("L") if img.mode != "L" else img
    if mode == "equalize":
        return ImageOps.equalize(gray)
    if mode == "autocontrast":
        return ImageOps.autocontrast(gray, cutoff=2)
    return img


def _load_phash(args):
    """加载单张图并计算 pHash。"""
    path, preprocess = args
    if not path.is_file():
        return None
    try:
        img = Image.open(path).convert("L")
        img = _normalize_brightness(img, preprocess)
        return imagehash.phash(img, hash_size=PHASH_SIZE)
    except Exception:
        return None


def _load_dhash(args):
    """加载单张图并计算 dHash（梯度哈希，抗亮度变化）。"""
    path, preprocess = args
    if not path.is_file():
        return None
    try:
        img = Image.open(path).convert("L")
        img = _normalize_brightness(img, preprocess)
        return imagehash.dhash(img, hash_size=PHASH_SIZE)
    except Exception:
        return None


def _vector_similarity_corr(va: np.ndarray, vb: np.ndarray) -> float:
    """两向量相关系数。"""
    if va is None or vb is None:
        return 0.0
    if va.std() == 0 and vb.std() == 0:
        return 1.0 if np.allclose(va, vb) else 0.0
    if va.std() == 0 or vb.std() == 0:
        return 0.0
    return float(np.corrcoef(va, vb)[0, 1])


def _hash_similarity(h1, h2) -> float:
    """哈希汉明距离转相似度：1 - (距离/位数)，范围 [0,1]。"""
    if h1 is None or h2 is None:
        return 0.0
    d = h1 - h2  # 汉明距离
    max_bits = int(h1.hash.size) if hasattr(h1, "hash") else 64
    return 1.0 - (d / max_bits)


def _compute_quality(path: Path, use_sobel: bool = True, use_saturation: bool = True, denoise: bool = True) -> dict:
    """
    计算单张图清晰度/质量得分。失败返回 None。
    - laplacian_var: 拉普拉斯方差，越高越清晰
    - sobel_mean: Sobel 梯度幅度均值，动漫线条越结实越高
    - saturation_mean: 饱和度均值，色彩丰富度
    - quality_score: 综合得分，用于组内排序
    """
    if not path.is_file():
        return None
    try:
        img = cv2.imread(str(path))
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 可选：轻微中值滤波降噪后再算梯度（抗噪点）
        if denoise:
            gray = cv2.medianBlur(gray, 3)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        result = {"laplacian_var": round(laplacian_var, 2)}
        if use_sobel:
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mean = float(np.mean(sobel_x**2 + sobel_y**2))
            result["sobel_mean"] = round(sobel_mean, 2)
        else:
            sobel_mean = 0.0
        if use_saturation:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation_mean = float(np.mean(hsv[:, :, 1]))
            result["saturation_mean"] = round(saturation_mean, 2)
        else:
            saturation_mean = 0.0
        # 综合得分：拉普拉斯为主 + Sobel 辅助（动漫线条）+ 饱和度微调
        quality_score = laplacian_var + 0.0001 * sobel_mean + 0.01 * saturation_mean
        result["quality_score"] = round(quality_score, 2)
        return result
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="连续相似帧分组，为 metadata 每条增加 group_id")
    parser.add_argument("--json", type=Path, default=DEFAULT_METADATA, help="metadata.json 路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 JSON 路径")
    parser.add_argument("--method", choices=["phash", "dhash", "corr"], default="phash",
                        help="相似度算法：phash 抗平移，dhash 抗亮度变化，corr 灰度相关系数")
    parser.add_argument("--preprocess", choices=["none", "equalize", "autocontrast"], default="equalize",
                        help="phash/dhash 前亮度归一化：equalize 直方图均衡，autocontrast 自动对比度，减轻整体变白影响")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="相邻帧相似度 >= 此值归为同一组（0~1）")
    parser.add_argument("--size", type=int, default=64, help="corr 方法时缩小边长（像素）")
    parser.add_argument("--limit", type=int, default=0, help="仅取前 N 条测试（0=全部）")
    parser.add_argument("--vis-dir", default=VIS_DIR, help="分组可视化输出目录（单目录，文件名加 group_id 前缀）")
    parser.add_argument("--no-vis", action="store_true", help="不复制图片到 vis-dir")
    parser.add_argument("--no-quality", action="store_true", help="不计算质量得分与组内最佳")
    parser.add_argument("--no-sobel", action="store_true", help="质量计算时不用 Sobel 梯度")
    parser.add_argument("--no-saturation", action="store_true", help="质量计算时不用饱和度")
    parser.add_argument("--no-denoise", action="store_true", help="Sobel 前不做中值滤波")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="预加载与复制时的线程池大小")
    args = parser.parse_args()

    if args.method in ("phash", "dhash") and imagehash is None:
        raise SystemExit("phash/dhash 需要 imagehash 库，请安装: pip install imagehash")

    base_dir = args.json.resolve().parent
    if not args.json.is_file():
        raise SystemExit(f"JSON 不存在: {args.json}")

    with open(args.json, "r", encoding="utf-8") as f:
        records = json.load(f)

    if args.limit > 0:
        records = records[: args.limit]
        print(f"仅处理前 {len(records)} 条")

    if not records:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print("无记录，已写出空列表")
        return

    paths = [base_dir / r["image_path"] for r in records]
    preprocess = args.preprocess if args.preprocess != "none" else "none"

    if args.method == "phash":
        load_args = [(p, preprocess) for p in paths]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            features = list(tqdm(
                ex.map(_load_phash, load_args),
                total=len(paths),
                desc="预加载 pHash",
                unit="张",
            ))
        sim_fn = _hash_similarity
    elif args.method == "dhash":
        load_args = [(p, preprocess) for p in paths]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            features = list(tqdm(
                ex.map(_load_dhash, load_args),
                total=len(paths),
                desc="预加载 dHash",
                unit="张",
            ))
        sim_fn = _hash_similarity
    else:
        size = (args.size, args.size)
        load_args = [(p, size) for p in paths]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            features = list(tqdm(
                ex.map(_load_thumb_corr, load_args),
                total=len(load_args),
                desc="预加载缩略图",
                unit="张",
            ))
        sim_fn = _vector_similarity_corr

    group_id = 0
    records[0]["group_id"] = group_id
    records[0]["similarity_to_prev"] = None  # 第一张无前一张

    for i in tqdm(range(1, len(records)), desc="连续帧分组", unit="张"):
        sim = sim_fn(features[i - 1], features[i])
        records[i]["similarity_to_prev"] = round(sim, 4)
        if sim >= args.threshold:
            records[i]["group_id"] = group_id
        else:
            group_id += 1
            records[i]["group_id"] = group_id

    # 计算质量得分，选出每组最佳
    if not args.no_quality:
        def _quality_task(idx):
            r = records[idx]
            p = base_dir / r["image_path"]
            q = _compute_quality(
                p,
                use_sobel=not args.no_sobel,
                use_saturation=not args.no_saturation,
                denoise=not args.no_denoise,
            )
            return idx, q

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            quality_results = list(tqdm(
                ex.map(_quality_task, range(len(records))),
                total=len(records),
                desc="计算质量得分",
                unit="张",
            ))
        for idx, q in quality_results:
            if q:
                records[idx].update(q)
                records[idx]["is_best_in_group"] = False  # 先设默认
            else:
                records[idx]["quality_score"] = 0.0
                records[idx]["is_best_in_group"] = False
        # 每组选质量最高的为最佳
        group_indices = defaultdict(list)
        for i, r in enumerate(records):
            group_indices[r["group_id"]].append(i)
        n_best = 0
        for gid, indices in group_indices.items():
            best_idx = max(indices, key=lambda i: records[i].get("quality_score", 0.0))
            records[best_idx]["is_best_in_group"] = True
            n_best += 1
        print(f"已选出 {n_best} 张组内最佳图（is_best_in_group=true）")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    n_groups = group_id + 1
    counts = {}
    for r in records:
        g = r["group_id"]
        counts[g] = counts.get(g, 0) + 1
    avg = sum(counts.values()) / len(counts) if counts else 0
    print(f"完成: 共 {len(records)} 条，{n_groups} 个组，平均每组 {avg:.1f} 张")
    print(f"已保存到 {args.output}")

    # 分组可视化：复制到 group/，文件名加 group_id_ 前缀，如 96_anime-screenshot-v2-632_1317.webp
    if not args.no_vis:
        vis_root = base_dir / args.vis_dir
        if vis_root.exists():
            shutil.rmtree(vis_root)
        vis_root.mkdir(parents=True, exist_ok=True)
        copy_tasks = []
        for r in records:
            src = (base_dir / r["image_path"]).resolve()
            if src.is_file():
                name = Path(r["image_path"]).name
                dst = vis_root / f"{r['group_id']}_{name}"
                copy_tasks.append((src, dst))
        if copy_tasks:
            def _copy_one(args):
                shutil.copy2(args[0], args[1])
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                list(tqdm(
                    ex.map(_copy_one, copy_tasks),
                    total=len(copy_tasks),
                    desc="复制到 group 目录",
                    unit="张",
                ))
        print(f"已复制到 {vis_root}，共 {len(copy_tasks)} 张（文件名含 group_id 前缀）")


if __name__ == "__main__":
    main()
