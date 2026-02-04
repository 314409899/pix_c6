"""
解压 zip3.tar 到 image 目录，并生成 metadata.json：
每条记录包含 image_path、image_name。
"""
import json
import os
import tarfile
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TAR = os.path.join(SCRIPT_DIR, "zip3.tar")
IMAGE_DIR = "image"
METADATA_JSON = "metadata.json"

# 视为图片的扩展名
IMAGE_EXT = {".webp", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif"}


def main():
    parser = argparse.ArgumentParser(description="解压 tar 到 image 并生成 metadata.json")
    parser.add_argument("--tar", default=DEFAULT_TAR, help="zip3.tar 路径")
    parser.add_argument("--out-dir", default=IMAGE_DIR, help="解压目标目录，默认 image")
    parser.add_argument("--metadata", default=METADATA_JSON, help="输出的 metadata 文件名")
    args = parser.parse_args()

    tar_path = os.path.abspath(args.tar)
    out_dir = os.path.abspath(args.out_dir)
    base_dir = os.path.dirname(tar_path)
    metadata_path = os.path.join(base_dir, args.metadata)

    if not os.path.isfile(tar_path):
        raise SystemExit(f"Tar 文件不存在: {tar_path}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"解压 {tar_path} -> {out_dir} ...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(out_dir)
    print("解压完成。")

    records = []
    for root, _dirs, files in os.walk(out_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXT:
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, base_dir)
            rel_slash = rel.replace("\\", "/")
            records.append({
                "image_path": rel_slash,
                "image_name": name,
            })

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"已生成 {metadata_path}，共 {len(records)} 条图片记录。")


if __name__ == "__main__":
    main()
