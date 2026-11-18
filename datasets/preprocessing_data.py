"""Tiền xử lý Oxford-IIIT Pet: kiểm tra dữ liệu và tạo phân chia train/val/test.

Dùng phân chia chính thức của bộ dữ liệu: ``trainval.txt`` (3.680 ảnh) được chia
tiếp 80/20 theo lớp thành train/val, ``test.txt`` (3.669 ảnh) giữ nguyên làm test.
Mỗi dòng trong file phân chia có dạng::

    <stem> <class_id 0..36> <species 1=cat,2=dog>

Chạy độc lập::

    python -m datasets.preprocessing_data --root datasets/Oxford-IIIT_Pet_Dataset
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.utils import REPO_ROOT, save_json  # noqa: E402

NUM_CLASSES = 37
SPLIT_FILES = ("train", "val", "test")


class PetEntry:
    __slots__ = ("stem", "label", "species")

    def __init__(self, stem: str, label: int, species: int) -> None:
        self.stem = stem
        self.label = label
        self.species = species

    def __repr__(self) -> str:  # pragma: no cover
        return f"PetEntry({self.stem}, label={self.label}, species={self.species})"


def _resolve_root(root: Path | str) -> Path:
    root = Path(root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root


def read_annotation_list(path: Path) -> List[PetEntry]:
    """Đọc ``list.txt`` / ``trainval.txt`` / ``test.txt`` của Oxford-IIIT Pet."""
    entries: List[PetEntry] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            stem, class_id, species = parts[0], int(parts[1]), int(parts[2])
            # CLASS-ID trong file là 1..37 -> chuyển về 0..36.
            entries.append(PetEntry(stem, class_id - 1, species))
    return entries


def class_names_from_entries(entries: Sequence[PetEntry]) -> List[str]:
    """Tên lớp = phần tiền tố của tên ảnh (``Abyssinian_100`` -> ``Abyssinian``)."""
    names: Dict[int, str] = {}
    for entry in entries:
        names.setdefault(entry.label, entry.stem.rsplit("_", 1)[0])
    return [names[i] for i in sorted(names)]


def filter_existing(entries: Sequence[PetEntry], root: Path, verbose: bool = True) -> List[PetEntry]:
    """Bỏ các mẫu thiếu ảnh hoặc thiếu trimap."""
    images_dir = root / "images"
    trimaps_dir = root / "annotations" / "trimaps"
    kept, missing = [], []
    for entry in entries:
        has_image = (images_dir / f"{entry.stem}.jpg").exists()
        has_trimap = (trimaps_dir / f"{entry.stem}.png").exists()
        if has_image and has_trimap:
            kept.append(entry)
        else:
            missing.append(entry.stem)
    if verbose and missing:
        print(f"[preprocess] bỏ {len(missing)} mẫu thiếu ảnh/trimap, ví dụ: {missing[:5]}")
    return kept


def stratified_split(
    entries: Sequence[PetEntry], val_ratio: float, seed: int
) -> Tuple[List[PetEntry], List[PetEntry]]:
    """Chia theo lớp (stratified) để mỗi giống xuất hiện trong cả train và val."""
    by_class: Dict[int, List[PetEntry]] = defaultdict(list)
    for entry in entries:
        by_class[entry.label].append(entry)

    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(by_class):
        items = sorted(by_class[label], key=lambda e: e.stem)
        rng.shuffle(items)
        n_val = max(1, int(round(val_ratio * len(items))))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    train.sort(key=lambda e: e.stem)
    val.sort(key=lambda e: e.stem)
    return train, val


def write_split(entries: Sequence[PetEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(f"{entry.stem} {entry.label} {entry.species}\n")


def load_split(path: Path | str) -> List[PetEntry]:
    entries: List[PetEntry] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stem, label, species = line.split()
            entries.append(PetEntry(stem, int(label), int(species)))
    return entries


def build_splits(
    root: Path | str,
    split_dir: Path | str | None = None,
    val_ratio: float = 0.2,
    seed: int = 0,
    overwrite: bool = False,
    verbose: bool = True,
) -> Dict[str, Path]:
    """Tạo (nếu chưa có) ba file phân chia và trả về đường dẫn của chúng."""
    root = _resolve_root(root)
    split_dir = Path(split_dir) if split_dir is not None else root / "splits"
    if not split_dir.is_absolute():
        split_dir = REPO_ROOT / split_dir

    paths = {name: split_dir / f"{name}.txt" for name in SPLIT_FILES}
    if all(p.exists() for p in paths.values()) and not overwrite:
        return paths

    ann = root / "annotations"
    if not ann.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {ann}. Tải Oxford-IIIT Pet (images.tar.gz, annotations.tar.gz) "
            f"và giải nén vào {root}."
        )

    trainval = filter_existing(read_annotation_list(ann / "trainval.txt"), root, verbose)
    test = filter_existing(read_annotation_list(ann / "test.txt"), root, verbose)
    train, val = stratified_split(trainval, val_ratio=val_ratio, seed=seed)

    write_split(train, paths["train"])
    write_split(val, paths["val"])
    write_split(test, paths["test"])

    meta = {
        "root": str(root),
        "num_classes": NUM_CLASSES,
        "class_names": class_names_from_entries(trainval + test),
        "val_ratio": val_ratio,
        "split_seed": seed,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
    }
    save_json(meta, split_dir / "meta.json")

    if verbose:
        print(f"[preprocess] train={len(train)}  val={len(val)}  test={len(test)}")
        print(f"[preprocess] ghi vào {split_dir}")
    return paths


def summarize(root: Path | str, split_dir: Path | str | None = None) -> None:
    """In thống kê nhanh: số ảnh mỗi split, tỷ lệ tiền cảnh trung bình."""
    import numpy as np
    from PIL import Image

    root = _resolve_root(root)
    paths = build_splits(root, split_dir)
    for name, path in paths.items():
        entries = load_split(path)
        per_class = defaultdict(int)
        for entry in entries:
            per_class[entry.label] += 1
        counts = np.array(sorted(per_class.values()))
        print(f"{name:>5}: {len(entries)} ảnh, {len(per_class)} lớp, "
              f"ảnh/lớp min={counts.min()} max={counts.max()}")

    sample = load_split(paths["train"])[:50]
    ratios = []
    for entry in sample:
        trimap = np.array(Image.open(root / "annotations" / "trimaps" / f"{entry.stem}.png"))
        ratios.append(float((trimap == 1).mean()))
    print(f"tỷ lệ tiền cảnh (trimap==1) trên 50 ảnh đầu: trung bình {np.mean(ratios):.3f}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tiền xử lý Oxford-IIIT Pet")
    parser.add_argument("--root", default="datasets/Oxford-IIIT_Pet_Dataset")
    parser.add_argument("--split-dir", default=None, help="mặc định <root>/splits")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)

    build_splits(args.root, args.split_dir, args.val_ratio, args.seed, args.overwrite)
    if args.summarize:
        summarize(args.root, args.split_dir)


if __name__ == "__main__":
    main()
