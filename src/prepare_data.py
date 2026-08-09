"""Build data/train|val|test from the Kaggle tomato-disease archive.

Split strategy: PRESERVE the archive's own train/valid boundary.

    archive/train/<class>/*  ->  data/train/<class>/
    archive/valid/<class>/*  ->  data/val/<class>/   (50%, stratified)
                             ->  data/test/<class>/  (50%, stratified)

Why not a random 70/15/15 re-split of everything pooled together: this dataset
aggregates multiple sources (lab PlantVillage images plus field photos) and contains
near-duplicate images. Pooling and re-shuffling puts near-duplicates on both sides of
the split, so test accuracy measures memorisation rather than generalisation -- you get
a ~99% number that means nothing. Keeping the archive's boundary means test images were
never in the training pool.

Determinism matters and is not incidental. Training happens in Colab while the Streamlit
demo runs locally, so both machines must compute the SAME assignment -- otherwise images
the model trained on end up in the local test split and local evaluation silently
overstates performance. Hence: filenames are sorted before splitting, the RNG is seeded,
and nothing depends on filesystem iteration order. split_manifest.json records the
result so the two machines can be diffed.

Framework-agnostic: file copying only, no TensorFlow import.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# tf.io.decode_image -- which image_dataset_from_directory calls -- only understands these
# four. Pillow reads far more (WebP, TIFF, ...), so a file can pass a naive PIL check and
# still crash training with "Unknown image file format. One of JPEG, PNG, GIF, BMP
# required." This archive really does ship WebP payloads inside .jpg filenames, so the
# check below tests the DECODED FORMAT rather than trusting the extension.
TF_READABLE_FORMATS = {"JPEG", "PNG", "GIF", "BMP"}

# Directory names the archive might use for its held-out portion.
VALID_ALIASES = ("valid", "validation", "val", "test")
TRAIN_ALIASES = ("train", "training")


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def list_images(directory: Path) -> list[Path]:
    """Images directly inside `directory`, sorted for reproducibility.

    sorted() is what makes the split deterministic across machines -- iterdir()
    returns filesystem order, which differs between Colab and a local disk.
    """
    return sorted((p for p in directory.iterdir() if is_image(p)), key=lambda p: p.name)


def class_dirs(directory: Path) -> list[Path]:
    """Immediate subdirectories that contain at least one image."""
    subdirs = sorted((p for p in directory.iterdir() if p.is_dir()), key=lambda p: p.name)
    return [d for d in subdirs if any(is_image(p) for p in d.iterdir())]


def find_child(directory: Path, aliases: tuple[str, ...]) -> Path | None:
    """Case-insensitive lookup of a child directory matching one of `aliases`."""
    by_lower = {p.name.lower(): p for p in directory.iterdir() if p.is_dir()}
    for alias in aliases:
        if alias in by_lower:
            return by_lower[alias]
    return None


def detect_layout(src: Path) -> tuple[str, dict[str, Path]]:
    """Work out how the extracted archive is arranged.

    Returns ("split", {"train": ..., "valid": ...}) when the archive ships its own
    train/valid folders, or ("flat", {"all": ...}) when it is just <class>/*.jpg.

    Descends through redundant single-child wrappers first -- zip files frequently
    extract to <name>/<name>/... and hardcoding either shape breaks on the other.
    """
    cursor = src
    for _ in range(4):
        train = find_child(cursor, TRAIN_ALIASES)
        valid = find_child(cursor, VALID_ALIASES)
        if train is not None and valid is not None:
            return "split", {"train": train, "valid": valid}

        if class_dirs(cursor):
            return "flat", {"all": cursor}

        children = [p for p in cursor.iterdir() if p.is_dir()]
        if len(children) != 1:
            break
        cursor = children[0]

    raise SystemExit(
        f"Could not recognise the dataset layout under {src}.\n"
        "Expected either train/ + valid/ subfolders, or class folders containing images.\n"
        f"Found: {sorted(p.name for p in src.iterdir())[:15]}"
    )


def copy_all(images: list[Path], dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy2(img, dest / img.name)
    return len(images)


def inspect_image(path: Path) -> tuple[str, str | None]:
    """Classify one file as ("ok" | "transcode" | "broken", detected_format).

    "transcode" means the pixels are fine but the container is one TensorFlow cannot read
    (WebP here) -- recoverable by re-encoding. "broken" means it does not decode at all.

    PIL rather than TensorFlow so this module stays framework-agnostic; TF_READABLE_FORMATS
    encodes what TF accepts without importing it. verify() alone checks headers without
    decoding pixel data, so a reopen and full load() is needed to catch truncation.
    """
    try:
        from PIL import Image
    except ImportError:  # pillow absent -- skip rather than fail the whole prepare
        return "ok", None
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            fmt = img.format
            img.convert("RGB").load()
    except Exception:
        return "broken", None
    return ("ok" if fmt in TF_READABLE_FORMATS else "transcode"), fmt


def transcode_to_jpeg(path: Path) -> bool:
    """Re-encode a readable-but-wrong-container image as real JPEG, in place.

    Repairing beats deleting: removing the file would change that split's count and make
    this machine's totals disagree with the numbers a previous training run reported.
    Re-encoding keeps every count identical while making the file actually loadable.
    """
    try:
        from PIL import Image

        with Image.open(path) as img:
            rgb = img.convert("RGB")
            rgb.load()
        rgb.save(path, format="JPEG", quality=95)
        return True
    except Exception:
        return False


def verify_split(out: Path) -> tuple[list[Path], list[Path]]:
    """Repair or remove unusable images in the built splits.

    Returns (removed, transcoded).

    Runs AFTER splitting, deliberately. Filtering inside list_images() would change how
    many images each class has and therefore which ones halve() sends to val vs test --
    silently desyncing this split from the one a previously-trained model was built on.
    Acting on files afterwards leaves every other file's assignment untouched.
    """
    removed: list[Path] = []
    transcoded: list[Path] = []
    for split in ("train", "val", "test"):
        split_dir = out / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.rglob("*")):
            if not path.is_file():
                continue
            verdict, fmt = inspect_image(path)
            if verdict == "ok":
                continue
            if verdict == "transcode" and transcode_to_jpeg(path):
                transcoded.append(path)
                continue
            removed.append(path)
            path.unlink()
    return removed, transcoded


def halve(images: list[Path], seed: int, class_name: str) -> tuple[list[Path], list[Path]]:
    """Split one class's images 50/50 into (val, test), deterministically.

    Seeded per class so adding or removing a class does not reshuffle the others.
    """
    shuffled = list(images)
    random.Random(f"{seed}:{class_name}").shuffle(shuffled)
    cut = len(shuffled) // 2
    return shuffled[:cut], shuffled[cut:]


def print_table(counts: dict[str, dict[str, int]]) -> None:
    classes = sorted(counts)
    width = max((len(c) for c in classes), default=5)
    print(f"\n{'class'.ljust(width)}  {'train':>7} {'val':>6} {'test':>6} {'total':>7}")
    print("-" * (width + 30))
    totals = {"train": 0, "val": 0, "test": 0}
    for name in classes:
        row = counts[name]
        for split in totals:
            totals[split] += row.get(split, 0)
        line_total = sum(row.get(s, 0) for s in totals)
        print(
            f"{name.ljust(width)}  {row.get('train', 0):>7} "
            f"{row.get('val', 0):>6} {row.get('test', 0):>6} {line_total:>7}"
        )
    print("-" * (width + 30))
    grand = sum(totals.values())
    print(
        f"{'TOTAL'.ljust(width)}  {totals['train']:>7} "
        f"{totals['val']:>6} {totals['test']:>6} {grand:>7}"
    )
    print(f"\n{len(classes)} classes, {grand} images.")

    if grand:
        smallest = min(classes, key=lambda c: sum(counts[c].values()))
        largest = max(classes, key=lambda c: sum(counts[c].values()))
        lo, hi = sum(counts[smallest].values()), sum(counts[largest].values())
        if lo:
            print(
                f"Imbalance ratio {hi / lo:.1f}x "
                f"(largest '{largest}' {hi}, smallest '{smallest}' {lo}) "
                "-- training uses class_weight to compensate."
            )


def prepare(src: Path, out: Path, seed: int, force: bool, verify: bool = True) -> dict:
    if out.exists() and any(out.iterdir()):
        if not force:
            raise SystemExit(
                f"{out} already exists and is not empty. Re-run with --force to rebuild."
            )
        shutil.rmtree(out)

    layout, roots = detect_layout(src)
    print(f"Detected layout: {layout}")
    for label, path in roots.items():
        print(f"  {label}: {path}")

    counts: dict[str, dict[str, int]] = {}
    test_files: dict[str, list[str]] = {}

    if layout == "split":
        for class_dir in class_dirs(roots["train"]):
            name = class_dir.name
            counts.setdefault(name, {})["train"] = copy_all(
                list_images(class_dir), out / "train" / name
            )

        for class_dir in class_dirs(roots["valid"]):
            name = class_dir.name
            val_imgs, test_imgs = halve(list_images(class_dir), seed, name)
            row = counts.setdefault(name, {})
            row["val"] = copy_all(val_imgs, out / "val" / name)
            row["test"] = copy_all(test_imgs, out / "test" / name)
            test_files[name] = sorted(p.name for p in test_imgs)
    else:
        # Flat <class>/*.jpg -- no archive boundary to preserve, so fall back to a
        # seeded 70/15/15. The near-duplicate leakage caveat applies here; say so in
        # the README if this branch is what ran.
        print(
            "\nWARNING: flat layout means there is no archive train/valid boundary to\n"
            "preserve. Falling back to a seeded 70/15/15 split. Near-duplicate images\n"
            "may span splits, which can inflate test accuracy -- note this in the README."
        )
        for class_dir in class_dirs(roots["all"]):
            name = class_dir.name
            images = list_images(class_dir)
            random.Random(f"{seed}:{name}").shuffle(images)
            n = len(images)
            n_train, n_val = int(n * 0.70), int(n * 0.15)
            parts = {
                "train": images[:n_train],
                "val": images[n_train : n_train + n_val],
                "test": images[n_train + n_val :],
            }
            counts[name] = {
                split: copy_all(imgs, out / split / name) for split, imgs in parts.items()
            }
            test_files[name] = sorted(p.name for p in parts["test"])

    excluded: list[str] = []
    repaired: list[str] = []
    if verify:
        print("\nVerifying every copied image decodes (a minute or two)...")
        removed, transcoded = verify_split(out)
        if transcoded:
            print(
                f"Re-encoded {len(transcoded)} image(s) whose real format TensorFlow "
                "cannot read (WebP data in a .jpg filename):"
            )
            for path in transcoded:
                rel = path.relative_to(out)
                repaired.append(str(rel))
                print(f"  {rel}")
            print("Counts are unaffected -- these files were repaired, not dropped.")
        if removed:
            print(f"Removed {len(removed)} undecodable file(s):")
            for path in removed:
                rel = path.relative_to(out)
                excluded.append(str(rel))
                print(f"  {rel}")
                split, class_name = rel.parts[0], rel.parts[1]
                key = "val" if split == "val" else split
                if class_name in counts and key in counts[class_name]:
                    counts[class_name][key] -= 1
                if split == "test" and class_name in test_files:
                    test_files[class_name] = [
                        n for n in test_files[class_name] if n != path.name
                    ]
            print("Counts below exclude them.")
        if not removed and not transcoded:
            print("All images decode cleanly.")

    print_table(counts)

    manifest = {
        "layout": layout,
        "strategy": "preserve" if layout == "split" else "pooled_70_15_15",
        "seed": seed,
        "verified": verify,
        "excluded_undecodable": excluded,
        # Files repaired in place. Recorded separately from exclusions because these did
        # NOT change any count -- the split assignment is unaffected, so a manifest with
        # this key still compares equal to one without it on every field that matters.
        "repaired_transcoded": repaired,
        "class_names": sorted(counts),
        "counts": counts,
        "test_files": test_files,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {out / 'split_manifest.json'}")
    print(
        "Keep this file. Diff it against the Colab copy to confirm both machines\n"
        "produced the same split -- a mismatch invalidates local evaluation."
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--src", required=True, type=Path, help="extracted archive directory"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true", help="delete and rebuild an existing --out"
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="skip the decode check (faster, but undecodable images reach training)",
    )
    args = parser.parse_args(argv)

    if not args.src.is_dir():
        raise SystemExit(f"--src {args.src} is not a directory")

    prepare(args.src.resolve(), args.out.resolve(), args.seed, args.force, args.verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
