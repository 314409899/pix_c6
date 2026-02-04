"""
YOLO 动漫人物检测 + 头部检测 + 人脸检测，合并脚本。
图片读写使用线程池并行加速。
- mode=person：对 metadata_grouped.json 做人物检测，仅用 is_best_in_group=true 的图片，输出 person_path、bbox
- mode=head：对 uuid JSON 做头部检测，读取 person_path 对应图，输出 head_path、head_bbox
- mode=face：对 uuid JSON 做人脸检测，读取 head_path 对应图，输出 face_path、face_bbox
- mode=all：依次执行 person -> head -> face 全流程
"""
import json
import os
import random
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm

DEFAULT_WORKERS = 16


def _load_image(path):
    """线程安全：加载单张图片，失败返回 None"""
    return cv2.imread(str(path))


def _write_image(args):
    """线程安全：保存单张图片，(path, img) -> bool"""
    path, img = args
    return cv2.imwrite(str(path), img)

SCRIPT_DIR = Path(__file__).resolve().parent
ANIME_PERSON_DIR = SCRIPT_DIR / "anime_person"
ANIME_HEAD_DIR = SCRIPT_DIR / "anime_head"
ANIME_FACE_DIR = SCRIPT_DIR / "anime_face"
DEFAULT_PERSON_MODEL = ANIME_PERSON_DIR / "model.pt"
DEFAULT_HEAD_MODEL = ANIME_HEAD_DIR / "model.pt"
DEFAULT_FACE_MODEL = ANIME_FACE_DIR / "model.pt"

# person 模式
DEFAULT_JSON_PERSON = SCRIPT_DIR / "metadata_grouped.json"
PERSON_DIR = "crops/person"
OUT_JSON_PERSON = "metadata_crops.json"
DEFAULT_PERSON_CONF = 0.324

# head 模式
DEFAULT_JSON_HEAD = SCRIPT_DIR / "metadata_crops_uuid.json"
DEFAULT_OUT_JSON_HEAD = SCRIPT_DIR / "metadata_crops_uuid_heads.json"
HEAD_DIR = "crops/heads"
DEFAULT_HEAD_CONF = 0.458

# face 模式
DEFAULT_JSON_FACE = SCRIPT_DIR / "metadata_crops_uuid_heads.json"
DEFAULT_OUT_JSON_FACE = SCRIPT_DIR / "metadata_crops_uuid_heads_faces.json"
FACE_DIR = "crops/faces"
DEFAULT_FACE_CONF = 0.307


def _process_one_person(rec, im, result, out_dir, out_dir_name):
    """单张图人物检测：裁剪，返回 (record, 裁剪数, [(path, crop), ...])。person_path 替代 crop_path。"""
    image_path = rec["image_path"]
    image_name = rec["image_name"]
    stem = Path(image_name).stem
    persons = []
    writes = []
    boxes = result.boxes if result else None
    person_idx = 0
    if boxes is not None:
        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(im.shape[1], x2), min(im.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = im[y1:y2, x1:x2]
            person_name = f"{stem}_crop_{person_idx}.webp"
            person_path = out_dir / person_name
            writes.append((person_path, crop))
            rel = f"{out_dir_name}/{person_name}".replace("\\", "/")
            persons.append({"person_path": rel, "bbox": [x1, y1, x2, y2], "person_index": person_idx})
            person_idx += 1
    return (
        {"image_path": image_path, "image_name": image_name, "crops": persons},
        len(persons),
        writes,
    )


def run_person_mode(args):
    """人物区域检测：metadata_1000.json -> 输出或返回（crops 内为 person_path）。skip_write 时不写 JSON，返回 (output_records, total)"""
    base_dir = args.json.resolve().parent
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    skip_write = getattr(args, "skip_write", False)

    with open(args.json, "r", encoding="utf-8") as f:
        meta_list = json.load(f)
    # metadata_grouped.json：仅使用 is_best_in_group=true 的图片
    if meta_list and "is_best_in_group" in meta_list[0]:
        meta_list = [r for r in meta_list if r.get("is_best_in_group") is True]
        print(f"已筛选 is_best_in_group=true，共 {len(meta_list)} 张")
    if args.limit > 0:
        if args.seed is not None:
            random.seed(args.seed)
        n = min(args.limit, len(meta_list))
        meta_list = random.sample(meta_list, n)
        print(f"随机抽取 {n} 张进行处理")

    model = __import__("ultralytics").YOLO(str(args.model))
    device = args.device or ("cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    use_half = args.half and "cuda" in str(device)
    batch_size = max(1, args.batch_size)

    output_records = []
    total = 0
    batch_recs = []
    batch_paths = []
    workers = getattr(args, "workers", DEFAULT_WORKERS)

    def run_batch():
        nonlocal total
        if not batch_paths:
            return
        with ThreadPoolExecutor(max_workers=workers) as ex:
            batch_ims = list(ex.map(_load_image, batch_paths))
        valid = [(r, im) for r, im in zip(batch_recs, batch_ims) if im is not None]
        for rec, im in zip(batch_recs, batch_ims):
            if im is None:
                tqdm.write(f"跳过（无法读取）: {rec['image_path']}")
                output_records.append({"image_path": rec["image_path"], "image_name": rec["image_name"], "crops": []})
        if not valid:
            batch_recs.clear()
            batch_paths.clear()
            return
        recs, ims = zip(*valid)
        results = model.predict(list(ims), conf=args.conf, device=device, verbose=False, half=use_half)
        all_writes = []
        for rec, im, result in zip(recs, ims, results):
            record, n, writes = _process_one_person(rec, im, result, out_dir, args.out_dir)
            output_records.append(record)
            total += n
            all_writes.extend(writes)
        if all_writes:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_write_image, all_writes))
        batch_recs.clear()
        batch_paths.clear()

    pbar = tqdm(meta_list, desc="提取人物区域", unit="张")
    for rec in pbar:
        src_path = (base_dir / rec["image_path"]).resolve()
        if not src_path.is_file():
            tqdm.write(f"跳过（文件不存在）: {rec['image_path']}")
            output_records.append({"image_path": rec["image_path"], "image_name": rec["image_name"], "crops": []})
            continue
        batch_recs.append(rec)
        batch_paths.append(src_path)
        if len(batch_paths) >= batch_size:
            run_batch()
        pbar.set_postfix(crops=total)

    run_batch()

    if not skip_write:
        out_json_path = base_dir / getattr(args, "out_json", Path(OUT_JSON_PERSON).name)
        if isinstance(out_json_path, str):
            out_json_path = base_dir / out_json_path
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(output_records, f, ensure_ascii=False, indent=2)
        print(f"完成: 共处理 {len(meta_list)} 张图，提取 {total} 个人物区域 -> {out_dir}，索引 -> {out_json_path}")
    else:
        print(f"person: {total} 个")
    if skip_write:
        return output_records, total


def run_head_mode(args):
    """头部检测：uuid 数据（source_image_name_crop_i 为 key）-> 输出 head_path、head_bbox。input_data 时跳过读文件，skip_write 时不写 JSON"""
    input_data = getattr(args, "input_data", None)
    if input_data is not None:
        base_dir = getattr(args, "base_dir", Path("."))
        data = input_data
    else:
        base_dir = args.json.resolve().parent
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
    head_dir = base_dir / args.out_dir
    head_dir.mkdir(parents=True, exist_ok=True)
    skip_write = getattr(args, "skip_write", False)

    uuids = list(data.keys())
    if args.limit > 0:
        if args.seed is not None:
            random.seed(args.seed)
        n = min(args.limit, len(uuids))
        uuids = random.sample(uuids, n)
        print(f"随机抽取 {n} 条进行处理")

    model = __import__("ultralytics").YOLO(str(args.model))
    device = args.device or ("cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    use_half = args.half and "cuda" in str(device)
    batch_size = max(1, args.batch_size)

    out_data = {}
    n_with_head = 0
    batch_uuids = []
    batch_paths = []
    workers = getattr(args, "workers", DEFAULT_WORKERS)

    def run_batch():
        nonlocal n_with_head
        if not batch_paths:
            return
        with ThreadPoolExecutor(max_workers=workers) as ex:
            batch_ims = list(ex.map(_load_image, batch_paths))
        valid = [(u, im) for u, im in zip(batch_uuids, batch_ims) if im is not None]
        for uuid, im in zip(batch_uuids, batch_ims):
            if im is None:
                tqdm.write(f"跳过（无法读取）: {data[uuid]['person_path']}")
                rec = dict(data[uuid])
                rec["head_path"] = None
                rec["head_bbox"] = None
                out_data[uuid] = rec
        if not valid:
            batch_uuids.clear()
            batch_paths.clear()
            return
        uuids, ims = zip(*valid)
        results = model.predict(list(ims), conf=args.conf, device=device, verbose=False, half=use_half)
        head_writes = []
        for uuid, im, result in zip(uuids, ims, results):
            rec = dict(data[uuid])
            boxes = result.boxes if result else None
            if boxes is not None and len(boxes) > 0:
                best = 0
                best_conf = float(boxes.conf[0].cpu().numpy())
                for i in range(1, len(boxes)):
                    c = float(boxes.conf[i].cpu().numpy())
                    if c > best_conf:
                        best_conf, best = c, i
                xyxy = boxes.xyxy[best].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(im.shape[1], x2), min(im.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    crop = im[y1:y2, x1:x2]
                    stem = Path(rec["person_path"]).stem
                    head_name = f"{stem}_head.webp"
                    head_writes.append((head_dir / head_name, crop))
                    rec["head_path"] = f"{args.out_dir}/{head_name}".replace("\\", "/")
                    rec["head_bbox"] = [x1, y1, x2, y2]
                    n_with_head += 1
                else:
                    rec["head_path"] = None
                    rec["head_bbox"] = None
            else:
                rec["head_path"] = None
                rec["head_bbox"] = None
            out_data[uuid] = rec
        if head_writes:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_write_image, head_writes))
        batch_uuids.clear()
        batch_paths.clear()

    pbar = tqdm(uuids, desc="提取头部区域", unit="张")
    for uuid in pbar:
        rec = data[uuid]
        person_path = rec["person_path"]
        src_path = (base_dir / person_path).resolve()
        if not src_path.is_file():
            tqdm.write(f"跳过（文件不存在）: {person_path}")
            rec = dict(rec)
            rec["head_path"] = None
            rec["head_bbox"] = None
            out_data[uuid] = rec
            continue
        batch_uuids.append(uuid)
        batch_paths.append(src_path)
        if len(batch_paths) >= batch_size:
            run_batch()
        pbar.set_postfix(heads=n_with_head)

    run_batch()

    n_with_head = sum(1 for v in out_data.values() if v.get("head_path") is not None)
    if not skip_write:
        out_json_path = args.out_json.resolve()
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f"完成: 共 {len(out_data)} 条，检测到头部 {n_with_head} 条 -> {head_dir}，索引 -> {out_json_path}")
    else:
        print(f"head: {n_with_head} 个（共 {len(out_data)} 条）")
    if skip_write:
        return out_data


def run_face_mode(args):
    """人脸检测：uuid 数据（含 head_path）-> 输出 face_path、face_bbox。input_data 时跳过读文件"""
    input_data = getattr(args, "input_data", None)
    if input_data is not None:
        base_dir = getattr(args, "base_dir", Path("."))
        data = input_data
    else:
        base_dir = args.json.resolve().parent
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
    face_dir = base_dir / args.out_dir
    face_dir.mkdir(parents=True, exist_ok=True)

    uuids = list(data.keys())
    if args.limit > 0:
        if args.seed is not None:
            random.seed(args.seed)
        n = min(args.limit, len(uuids))
        uuids = random.sample(uuids, n)
        print(f"随机抽取 {n} 条进行处理")

    model = __import__("ultralytics").YOLO(str(args.model))
    device = args.device or ("cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    use_half = args.half and "cuda" in str(device)
    batch_size = max(1, args.batch_size)

    out_data = {}
    n_with_face = 0
    batch_uuids = []
    batch_paths = []
    workers = getattr(args, "workers", DEFAULT_WORKERS)

    def run_batch():
        nonlocal n_with_face
        if not batch_paths:
            return
        with ThreadPoolExecutor(max_workers=workers) as ex:
            batch_ims = list(ex.map(_load_image, batch_paths))
        valid = [(u, im) for u, im in zip(batch_uuids, batch_ims) if im is not None]
        for uuid, im in zip(batch_uuids, batch_ims):
            if im is None:
                tqdm.write(f"跳过（无法读取）: {data[uuid].get('head_path', uuid)}")
                rec = dict(data[uuid])
                rec["face_path"] = None
                rec["face_bbox"] = None
                out_data[uuid] = rec
        if not valid:
            batch_uuids.clear()
            batch_paths.clear()
            return
        uuids, ims = zip(*valid)
        results = model.predict(list(ims), conf=args.conf, device=device, verbose=False, half=use_half)
        face_writes = []
        for uuid, im, result in zip(uuids, ims, results):
            rec = dict(data[uuid])
            head_path = rec.get("head_path")
            if not head_path:
                rec["face_path"] = None
                rec["face_bbox"] = None
                out_data[uuid] = rec
                continue
            boxes = result.boxes if result else None
            if boxes is not None and len(boxes) > 0:
                best = 0
                best_conf = float(boxes.conf[0].cpu().numpy())
                for i in range(1, len(boxes)):
                    c = float(boxes.conf[i].cpu().numpy())
                    if c > best_conf:
                        best_conf, best = c, i
                xyxy = boxes.xyxy[best].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(im.shape[1], x2), min(im.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    crop = im[y1:y2, x1:x2]
                    stem = Path(rec["person_path"]).stem
                    face_name = f"{stem}_face.webp"
                    face_writes.append((face_dir / face_name, crop))
                    rec["face_path"] = f"{args.out_dir}/{face_name}".replace("\\", "/")
                    rec["face_bbox"] = [x1, y1, x2, y2]
                    n_with_face += 1
                else:
                    rec["face_path"] = None
                    rec["face_bbox"] = None
            else:
                rec["face_path"] = None
                rec["face_bbox"] = None
            out_data[uuid] = rec
        if face_writes:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_write_image, face_writes))
        batch_uuids.clear()
        batch_paths.clear()

    pbar = tqdm(uuids, desc="提取人脸区域", unit="张")
    for uuid in pbar:
        rec = data[uuid]
        head_path = rec.get("head_path")
        if not head_path:
            # tqdm.write(f"跳过（无 head_path）: {uuid}")
            rec = dict(rec)
            rec["face_path"] = None
            rec["face_bbox"] = None
            out_data[uuid] = rec
            continue
        src_path = (base_dir / head_path).resolve()
        if not src_path.is_file():
            tqdm.write(f"跳过（文件不存在）: {head_path}")
            rec = dict(rec)
            rec["face_path"] = None
            rec["face_bbox"] = None
            out_data[uuid] = rec
            continue
        batch_uuids.append(uuid)
        batch_paths.append(src_path)
        if len(batch_paths) >= batch_size:
            run_batch()
        pbar.set_postfix(faces=n_with_face)

    run_batch()

    n_with_face = sum(1 for v in out_data.values() if v.get("face_path") is not None)
    out_json_path = getattr(args, "out_json_path", None) or (getattr(args, "out_json", None) and Path(args.out_json).resolve())
    if out_json_path:
        with open(Path(out_json_path).resolve(), "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f"完成: 共 {len(out_data)} 条，检测到人脸 {n_with_face} 条 -> {face_dir}，索引 -> {out_json_path}")


def _crops_to_uuid(crops_data):
    """将 crops 列表转为 uuid dict（source_image_name_crop_i 为 key）"""
    by_uuid = {}
    for rec in crops_data:
        source_image_name = rec["image_name"]
        for i, crop in enumerate(rec.get("crops", [])):
            uuid_key = f"{source_image_name}_crop_{i}"
            by_uuid[uuid_key] = {
                "person_path": crop["person_path"],
                "bbox": crop["bbox"],
                "person_index": crop["person_index"],
                "source_image_path": rec["image_path"],
                "source_image_name": source_image_name,
            }
    return by_uuid


def run_all_mode(args):
    """全流程：person -> head -> face，不输出中间 JSON，最终写入 metadata_crops.json"""
    base_dir = args.json.resolve().parent
    final_output = base_dir / "metadata_crops.json"

    # 1. person（不写 JSON）
    print("=== 步骤 1/3: 人物检测 ===")
    workers = getattr(args, "workers", DEFAULT_WORKERS)
    person_args = argparse.Namespace(
        json=args.json,
        out_dir=PERSON_DIR,
        out_json=Path(OUT_JSON_PERSON).name,
        model=DEFAULT_PERSON_MODEL,
        conf=DEFAULT_PERSON_CONF,
        device=args.device,
        batch_size=args.batch_size,
        workers=workers,
        half=args.half,
        limit=args.limit,
        seed=args.seed,
        skip_write=True,
    )
    output_records, n_person = run_person_mode(person_args)

    # 2. crops -> uuid（内存）
    uuid_data = _crops_to_uuid(output_records)

    # 3. head（不写 JSON）
    print("=== 步骤 2/3: 头部检测 ===")
    head_args = argparse.Namespace(
        json=None,
        base_dir=base_dir,
        input_data=uuid_data,
        out_dir=HEAD_DIR,
        out_json=None,
        model=DEFAULT_HEAD_MODEL,
        conf=DEFAULT_HEAD_CONF,
        device=args.device,
        batch_size=args.batch_size,
        workers=workers,
        half=args.half,
        limit=0,
        seed=args.seed,
        skip_write=True,
    )
    head_data = run_head_mode(head_args)

    # 4. face（只写 metadata_crops.json）
    print("=== 步骤 3/3: 人脸检测 ===")
    face_args = argparse.Namespace(
        json=None,
        base_dir=base_dir,
        input_data=head_data,
        out_dir=FACE_DIR,
        out_json=None,
        out_json_path=final_output,
        model=DEFAULT_FACE_MODEL,
        conf=DEFAULT_FACE_CONF,
        device=args.device,
        batch_size=args.batch_size,
        workers=workers,
        half=args.half,
        limit=0,
        seed=args.seed,
    )
    run_face_mode(face_args)

    with open(final_output, "r", encoding="utf-8") as f:
        final_data = json.load(f)
    n_with_head = sum(1 for v in head_data.values() if v.get("head_path"))
    n_with_face = sum(1 for v in final_data.values() if v.get("face_path"))
    print(f"=== 全流程完成: person {n_person} 个, head {n_with_head} 个（共 {len(head_data)} 条）, face {n_with_face} 个（共 {len(final_data)} 条）-> {final_output} ===")


def main():
    parser = argparse.ArgumentParser(description="YOLO 动漫人物 + 头部 + 人脸检测（person -> head -> face）")
    parser.add_argument("mode", nargs="?", default="all", choices=["person", "head", "face", "all"], help="person/head/face 单步，all=全流程（默认）")
    parser.add_argument("--json", type=Path, help="输入 JSON（person 默认 metadata_1000.json，head/face 默认上一级输出）")
    parser.add_argument("--out-dir", help="输出目录")
    parser.add_argument("--out-json", type=Path, help="输出 JSON（head/face 模式）")
    parser.add_argument("--model", type=Path, help="模型路径")
    # parser.add_argument("--conf", type=float, help="置信度阈值")
    parser.add_argument("--device", default="", help="推理设备")
    parser.add_argument("--batch-size", type=int, default=16, help="批量推理张数")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="图片读写线程池大小")
    parser.add_argument("--half", action="store_true", help="GPU 半精度")
    parser.add_argument("--limit", type=int, default=0, help="随机抽取 N 条（0=全部）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("请安装: pip install ultralytics opencv-python")

    if args.mode == "all":
        args.json = args.json or DEFAULT_JSON_PERSON
        run_all_mode(args)
        return

    if args.mode == "person":
        args.json = args.json or DEFAULT_JSON_PERSON
        args.out_dir = args.out_dir or PERSON_DIR
        args.out_json = args.out_json or Path(OUT_JSON_PERSON).name
        args.model = args.model or DEFAULT_PERSON_MODEL
        args.conf = args.conf if args.conf is not None else DEFAULT_PERSON_CONF
        if not args.model.is_file():
            raise SystemExit(f"模型不存在: {args.model}")
        run_person_mode(args)
    elif args.mode == "head":
        args.json = args.json or DEFAULT_JSON_HEAD
        args.out_dir = args.out_dir or HEAD_DIR
        args.out_json = args.out_json or DEFAULT_OUT_JSON_HEAD
        args.model = args.model or DEFAULT_HEAD_MODEL
        args.conf = args.conf if args.conf is not None else DEFAULT_HEAD_CONF
        if not args.model.is_file():
            raise SystemExit(f"模型不存在: {args.model}")
        run_head_mode(args)
    else:  # face
        args.json = args.json or DEFAULT_JSON_FACE
        args.out_dir = args.out_dir or FACE_DIR
        args.out_json = args.out_json or DEFAULT_OUT_JSON_FACE
        args.model = args.model or DEFAULT_FACE_MODEL
        args.conf = args.conf if args.conf is not None else DEFAULT_FACE_CONF
        if not args.model.is_file():
            raise SystemExit(f"模型不存在: {args.model}")
        run_face_mode(args)


if __name__ == "__main__":
    main()
