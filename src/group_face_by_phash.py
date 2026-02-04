"""
对 metadata_crops.json 分别对 face_path 和 head_path 做 pHash 代表元分组。
输出 face_group_id、head_group_id，组内第一张图为代表，标记 first_in_face_group、first_in_head_group。
"""
import json
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import imagehash

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = SCRIPT_DIR / "metadata_crops.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "metadata_crops_grouped.json"
OUT_FACE_DIR = "group_face"
OUT_HEAD_DIR = "group_head"
DEFAULT_THRESHOLD = 0.85  # pHash 相似度 >= 此值归为同一组
DEFAULT_WORKERS = 32
PHASH_SIZE = 8


def _normalize_brightness(img: Image.Image, mode: str) -> Image.Image:
    if not mode or mode == "none":
        return img
    gray = img.convert("L") if img.mode != "L" else img
    if mode == "equalize":
        return ImageOps.equalize(gray)
    if mode == "autocontrast":
        return ImageOps.autocontrast(gray, cutoff=2)
    return img


def _load_phash(args):
    path, preprocess = args
    if not path.is_file():
        return None
    try:
        img = Image.open(path).convert("L")
        img = _normalize_brightness(img, preprocess)
        return imagehash.phash(img, hash_size=PHASH_SIZE)
    except Exception:
        return None


def _run_representative_grouping(H: np.ndarray, max_hamming: int):
    """代表元分组，返回 (idx_to_gid, first_in_group_set)。"""
    n = H.shape[0]
    groups = []
    idx_to_gid = [-1] * n
    for i in tqdm(range(n), desc="代表元分组", unit="张", mininterval=0.5, leave=False):
        dist_to_reps = np.bitwise_xor(H[i], H[[g[0] for g in groups]]).sum(axis=1) if groups else np.array([])
        matched = np.where(dist_to_reps <= max_hamming)[0] if len(dist_to_reps) > 0 else np.array([])
        if len(matched) > 0:
            gid = int(matched[np.argmin(dist_to_reps[matched])])
            groups[gid][1].append(i)
            idx_to_gid[i] = gid
        else:
            gid = len(groups)
            groups.append((i, [i]))
            idx_to_gid[i] = gid
    groups_sorted = sorted(groups, key=lambda g: -len(g[1]))
    rep_to_old = {g[0]: i for i, g in enumerate(groups)}
    old_to_new = {rep_to_old[g[0]]: new_gid for new_gid, g in enumerate(groups_sorted)}
    idx_to_gid = [old_to_new[idx_to_gid[i]] for i in range(n)]
    first_in_group = {g[0] for g in groups_sorted}  # 代表元即组内第一张
    return idx_to_gid, first_in_group, len(groups_sorted)


def _hash_to_bits(h) -> np.ndarray:
    """将 imagehash 转为 (64,) uint8 向量（0/1），None 用全 0。"""
    if h is None or not hasattr(h, "hash"):
        return np.zeros(64, dtype=np.uint8)
    arr = np.asarray(h.hash, dtype=np.uint8).flatten()
    bits = (arr > 0).astype(np.uint8)
    if bits.size >= 64:
        return bits[:64]
    out = np.zeros(64, dtype=np.uint8)
    out[: bits.size] = bits
    return out


def main():
    parser = argparse.ArgumentParser(description="metadata_crops face 按 pHash 全局分组")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="metadata_crops.json 路径")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help="输出 JSON 路径")
    parser.add_argument("--out-face-dir", default=OUT_FACE_DIR, help="face 图分组输出目录")
    parser.add_argument("--out-head-dir", default=OUT_HEAD_DIR, help="head 图分组输出目录")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="相似度 >= 此值归为同一组")
    parser.add_argument("--preprocess", choices=["none", "equalize", "autocontrast"], default="equalize",
                        help="pHash 前亮度归一化")
    parser.add_argument("--limit", type=int, default=0, help="仅取前 N 条测试（0=全部）")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="预加载线程数")
    args = parser.parse_args()

    base_dir = args.json.resolve().parent
    if not args.json.is_file():
        raise SystemExit(f"JSON 不存在: {args.json}")

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_uuids = list(data.keys())
    if args.limit > 0:
        all_uuids = all_uuids[: args.limit]
        print(f"仅处理前 {len(all_uuids)} 条")

    face_uuids = [k for k in all_uuids if data[k].get("face_path")]
    head_uuids = [k for k in all_uuids if data[k].get("head_path")]
    preprocess = args.preprocess if args.preprocess != "none" else "none"
    max_hamming = int(64 * (1 - args.threshold))

    # 初始化所有记录的 group 字段
    for uuid in data:
        rec = data[uuid]
        rec["face_group_id"] = None
        rec["head_group_id"] = None
        rec["first_in_face_group"] = False
        rec["first_in_head_group"] = False

    # === Face 分组 ===
    if face_uuids:
        load_args = [(base_dir / data[u]["face_path"], preprocess) for u in face_uuids]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            face_hashes = list(tqdm(
                ex.map(_load_phash, load_args),
                total=len(load_args),
                desc="预加载 face pHash",
                unit="张",
                mininterval=0.5,
            ))
        H_face = np.stack([_hash_to_bits(h) for h in face_hashes], axis=0)
        face_idx_to_gid, face_first_set, n_face_groups = _run_representative_grouping(H_face, max_hamming)
        for i, uuid in enumerate(face_uuids):
            data[uuid]["face_group_id"] = face_idx_to_gid[i]
            data[uuid]["first_in_face_group"] = i in face_first_set
        print(f"Face 分组: {len(face_uuids)} 张 -> {n_face_groups} 个组")

    # === Head 分组 ===
    if head_uuids:
        load_args = [(base_dir / data[u]["head_path"], preprocess) for u in head_uuids]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            head_hashes = list(tqdm(
                ex.map(_load_phash, load_args),
                total=len(load_args),
                desc="预加载 head pHash",
                unit="张",
                mininterval=0.5,
            ))
        H_head = np.stack([_hash_to_bits(h) for h in head_hashes], axis=0)
        head_idx_to_gid, head_first_set, n_head_groups = _run_representative_grouping(H_head, max_hamming)
        for i, uuid in enumerate(head_uuids):
            data[uuid]["head_group_id"] = head_idx_to_gid[i]
            data[uuid]["first_in_head_group"] = i in head_first_set
        print(f"Head 分组: {len(head_uuids)} 张 -> {n_head_groups} 个组")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存 JSON 到 {args.output}")

    # 复制 face 到 group_face/
    if face_uuids:
        out_face = base_dir / args.out_face_dir
        if out_face.exists():
            shutil.rmtree(out_face)
        out_face.mkdir(parents=True, exist_ok=True)
        for i, uuid in enumerate(face_uuids):
            src = (base_dir / data[uuid]["face_path"]).resolve()
            if src.is_file():
                dst = out_face / f"{face_idx_to_gid[i]}_{Path(data[uuid]['face_path']).name}"
                shutil.copy2(src, dst)
        print(f"已复制 face 到 {out_face}")

    # 复制 head 到 group_head/
    if head_uuids:
        out_head = base_dir / args.out_head_dir
        if out_head.exists():
            shutil.rmtree(out_head)
        out_head.mkdir(parents=True, exist_ok=True)
        for i, uuid in enumerate(head_uuids):
            src = (base_dir / data[uuid]["head_path"]).resolve()
            if src.is_file():
                dst = out_head / f"{head_idx_to_gid[i]}_{Path(data[uuid]['head_path']).name}"
                shutil.copy2(src, dst)
        print(f"已复制 head 到 {out_head}")


if __name__ == "__main__":
    main()
