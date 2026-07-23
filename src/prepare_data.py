#!/usr/bin/env python3
"""
Galaxy-X-os -- src/prepare_data.py
==================================

SINGLE ENTRYPOINT that assembles the full 5-class dataset consumed by
``src/dataset.py`` at ``data/processed/{train,val,test}/<class>/*.png``.

Strategy: REAL-FIRST, SAFE-FALLBACK, IDEMPOTENT
-----------------------------------------------
* spiral_galaxy + elliptical_galaxy
    -> REAL: Galaxy10 (DECaLS preferred, SDSS fallback) via ``astroNN``.
       No API key. astroNN downloads the ~200MB .h5 on first use.
       Galaxy10's 10 morphology classes are mapped to spiral vs elliptical.
    -> FALLBACK: clearly-labelled procedural morphology generators.

* nebula + star_cluster + planetary
    -> REAL: documented Kaggle datasets, only if credentials exist
       (~/.kaggle/kaggle.json  or  KAGGLE_USERNAME/KAGGLE_KEY env vars).
       Uses ``kagglehub`` if importable, else the ``kaggle`` CLI.
    -> FALLBACK: clearly-labelled procedural generators (patterns lifted
       from ``src/generate_from_real_data.py``).

Whatever actually succeeds is recorded, per class, in
``data/processed/DATA_MANIFEST.json`` -- the honesty record the report cites.

Guarantees
----------
* Balanced to ``--per-class`` images per class (default 500).
* DISJOINT stratified 80/10/10 split (StratifiedShuffleSplit), verified by
  an MD5 content-hash leakage check that asserts NO image appears in >1 split.
* Emits ``data/processed/class_weights.json`` (inverse-frequency, keyed by the
  exact CLASSES names) and a printed summary table.
* Re-runnable: rebuilds the split directories from scratch each run.

Usage
-----
    python src/prepare_data.py                    # real-first, per-class=500
    python src/prepare_data.py --per-class 300
    python src/prepare_data.py --real-only        # disable procedural fallback
    python src/prepare_data.py --seed 7 --image-size 224
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import shutil
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Make sibling modules importable when run as ``python src/prepare_data.py``
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import CLASSES  # exact 5 folder/label names, index 0..4  # noqa: E402

# Optional: reuse procedural patterns already in the repo (nebula / elliptical).
try:
    from generate_from_real_data import (
        generate_nebula_images as _repo_nebula,
        generate_elliptical_images as _repo_elliptical,
    )
    _HAVE_REPO_GENERATORS = True
except Exception:  # pragma: no cover - defensive
    _HAVE_REPO_GENERATORS = False

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}

# Image = uint8 RGB HxWx3 numpy array throughout this module.
Array = np.ndarray

# ---------------------------------------------------------------------------
# Galaxy10 -> (spiral / elliptical) label maps
# ---------------------------------------------------------------------------
# Galaxy10 DECaLS 10-class scheme (astroNN "galaxy10"):
#   0 Disturbed, 1 Merging, 2 Round Smooth, 3 In-between Round Smooth,
#   4 Cigar Shaped Smooth, 5 Barred Spiral, 6 Unbarred Tight Spiral,
#   7 Unbarred Loose Spiral, 8 Edge-on w/o Bulge, 9 Edge-on w/ Bulge
GALAXY10_DECALS_MAP = {
    "elliptical_galaxy": {2, 3, 4},        # round / in-between / cigar smooth
    "spiral_galaxy": {5, 6, 7, 8, 9},      # barred/unbarred spirals + edge-on disks
    # 0 Disturbed, 1 Merging -> ambiguous, intentionally skipped
}
# Galaxy10 SDSS 10-class scheme (astroNN "galaxy10sdss"):
#   0 Disk Face-on No Spiral, 1 Smooth Completely Round, 2 Smooth In-between,
#   3 Smooth Cigar, 4 Disk Edge-on Rounded Bulge, 5 Disk Edge-on Boxy Bulge,
#   6 Disk Edge-on No Bulge, 7 Disk Face-on Tight Spiral,
#   8 Disk Face-on Medium Spiral, 9 Disk Face-on Loose Spiral
GALAXY10_SDSS_MAP = {
    "elliptical_galaxy": {1, 2, 3},        # smooth round / in-between / cigar
    "spiral_galaxy": {0, 4, 5, 6, 7, 8, 9},  # disks + spirals + edge-on
}

# Documented Kaggle datasets for the non-galaxy classes (real-first).
KAGGLE_DATASETS = {
    "nebula": ("fedesoriano/deep-space-images",
               ["nebula", "nebulae", "gas", "cloud"]),
    "star_cluster": ("fedesoriano/deep-space-images",
                     ["cluster", "globular", "open_cluster", "star_cluster"]),
    "planetary": ("brsdincer/planetary-solar-system-objects",
                  ["planet", "planetary", "moon", "satellite", "mars",
                   "jupiter", "saturn", "lunar"]),
}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def to_rgb_uint8(arr: Array, image_size: int) -> Array:
    """Coerce any array to an RGB uint8 image resized to (image_size, image_size)."""
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        a = a - a.min()
        m = a.max()
        if m > 0:
            a = a / m * 255.0
        arr = a.astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.LANCZOS)
    return np.array(img)


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def md5_array(arr: Array) -> str:
    return md5_bytes(np.ascontiguousarray(arr).tobytes())


def dedupe(arrs: List[Array]) -> List[Array]:
    """Drop exact-duplicate images (keeps split disjointness honest)."""
    seen, out = set(), []
    for a in arrs:
        h = md5_array(a)
        if h not in seen:
            seen.add(h)
            out.append(a)
    return out


# ---------------------------------------------------------------------------
# REAL source 1: Galaxy10 via astroNN  (spiral + elliptical)
# ---------------------------------------------------------------------------

def _load_galaxy10_raw() -> Optional[Tuple[Array, Array, Dict[str, set], str]]:
    """Try DECaLS then SDSS. Returns (images, labels, map, source_name) or None."""
    try:
        import astroNN  # noqa: F401
    except Exception as e:
        print(f"  [galaxy10] astroNN not importable ({e}); will use fallback.")
        return None

    # DECaLS (preferred) then SDSS. astroNN's public API has moved around across
    # versions, so try several documented import paths defensively.
    attempts = [
        ("Galaxy10 DECaLS (astroNN)", GALAXY10_DECALS_MAP,
         [("astroNN.datasets", "load_galaxy10"),
          ("astroNN.datasets.galaxy10", "load_data")]),
        ("Galaxy10 SDSS (astroNN)", GALAXY10_SDSS_MAP,
         [("astroNN.datasets", "load_galaxy10sdss"),
          ("astroNN.datasets.galaxy10sdss", "load_data")]),
    ]
    for source_name, cmap, loaders in attempts:
        for module_name, func_name in loaders:
            try:
                mod = __import__(module_name, fromlist=[func_name])
                loader = getattr(mod, func_name)
                print(f"  [galaxy10] loading {source_name} via "
                      f"{module_name}.{func_name} (first run downloads the .h5)...")
                images, labels = loader()
                images = np.asarray(images)
                labels = np.asarray(labels).astype(int).ravel()
                print(f"  [galaxy10] loaded {len(images)} images from {source_name}.")
                return images, labels, cmap, source_name
            except Exception as e:
                print(f"  [galaxy10] {module_name}.{func_name} failed ({e}).")
    return None


def get_galaxy10(per_class: int, image_size: int, rng: random.Random
                 ) -> Dict[str, Tuple[List[Array], str]]:
    """Return {spiral/elliptical: (images, source_label)} from Galaxy10, or {}."""
    loaded = _load_galaxy10_raw()
    if loaded is None:
        return {}
    images, labels, cmap, source_name = loaded

    url = ("https://astronn.readthedocs.io/en/latest/galaxy10.html "
           "(DECaLS) / galaxy10sdss.html (SDSS)")
    result: Dict[str, Tuple[List[Array], str]] = {}
    try:
        for cls, orig_labels in cmap.items():
            idx = np.where(np.isin(labels, list(orig_labels)))[0].tolist()
            rng.shuffle(idx)
            picked = idx[: per_class]
            arrs = [to_rgb_uint8(images[i], image_size) for i in picked]
            arrs = dedupe(arrs)
            if arrs:
                result[cls] = (arrs, f"{source_name} | {url}")
                print(f"  [galaxy10] {cls}: {len(arrs)} real images "
                      f"(mapped from labels {sorted(orig_labels)}).")
    finally:
        del images, labels  # free the large arrays promptly
    return result


# ---------------------------------------------------------------------------
# REAL source 2: Kaggle  (nebula + star_cluster + planetary)
# ---------------------------------------------------------------------------

def kaggle_credentials_present() -> bool:
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return True
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def _download_kaggle_dataset(ref: str) -> Optional[Path]:
    """Download a Kaggle dataset via kagglehub (preferred) or the kaggle CLI."""
    # kagglehub returns a cached local path -- ideal and idempotent.
    try:
        import kagglehub
        path = kagglehub.dataset_download(ref)
        print(f"  [kaggle] kagglehub cached '{ref}' at {path}")
        return Path(path)
    except Exception as e:
        print(f"  [kaggle] kagglehub unavailable/failed for '{ref}' ({e}); "
              f"trying kaggle CLI.")

    dest = RAW_DIR / ref.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.rglob("*")):
        return dest
    import subprocess
    import zipfile
    try:
        rc = subprocess.run(
            ["kaggle", "datasets", "download", "-d", ref, "-p", str(dest), "-q"],
            capture_output=True, text=True,
        ).returncode
        if rc != 0:
            print(f"  [kaggle] CLI download failed for '{ref}'.")
            return None
        for zf in dest.glob("*.zip"):
            with zipfile.ZipFile(zf) as z:
                z.extractall(dest)
            zf.unlink()
        return dest
    except FileNotFoundError:
        print("  [kaggle] kaggle CLI not installed.")
        return None
    except Exception as e:
        print(f"  [kaggle] CLI error for '{ref}': {e}")
        return None


def _discover_class_images(root: Path, keywords: List[str], limit: int,
                           image_size: int) -> List[Array]:
    """Collect up to `limit` images under `root` whose path matches a keyword."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    kws = [k.lower() for k in keywords]
    hits: List[Path] = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and any(k in str(p).lower() for k in kws):
            hits.append(p)
    arrs: List[Array] = []
    for p in hits:
        if len(arrs) >= limit:
            break
        try:
            arrs.append(to_rgb_uint8(np.array(Image.open(p).convert("RGB")), image_size))
        except Exception:
            continue
    return dedupe(arrs)


def get_kaggle_classes(classes: List[str], per_class: int, image_size: int
                       ) -> Dict[str, Tuple[List[Array], str]]:
    """Return {class: (images, source_label)} for whatever Kaggle yields."""
    if not kaggle_credentials_present():
        print("  [kaggle] no credentials (~/.kaggle/kaggle.json or KAGGLE_* env); "
              "skipping real Kaggle download.")
        return {}

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: Dict[str, Path] = {}
    result: Dict[str, Tuple[List[Array], str]] = {}
    for cls in classes:
        if cls not in KAGGLE_DATASETS:
            continue
        ref, keywords = KAGGLE_DATASETS[cls]
        if ref not in downloaded:
            path = _download_kaggle_dataset(ref)
            if path is None:
                continue
            downloaded[ref] = path
        arrs = _discover_class_images(downloaded[ref], keywords, per_class, image_size)
        if arrs:
            result[cls] = (arrs, f"Kaggle:{ref} | https://www.kaggle.com/datasets/{ref}")
            print(f"  [kaggle] {cls}: {len(arrs)} real images from {ref}.")
        else:
            print(f"  [kaggle] {cls}: no matching images found in {ref}.")
    return result


# ---------------------------------------------------------------------------
# PROCEDURAL fallback generators (clearly labelled as such in the manifest)
# ---------------------------------------------------------------------------

def _proc_spiral(count: int, image_size: int, rng: np.random.Generator) -> List[Array]:
    """Procedural spiral galaxy: logarithmic spiral arms + bright bulge."""
    out = []
    for _ in range(count):
        s = image_size
        img = np.zeros((s, s, 3), dtype=np.float32)
        cy, cx = s / 2, s / 2
        Y, X = np.ogrid[:s, :s]
        dx, dy = (X - cx) / (s / 2), (Y - cy) / (s / 2)
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        n_arms = rng.integers(2, 4)
        twist = rng.uniform(4.0, 7.0)
        arm = np.cos(n_arms * theta - twist * np.log(r + 0.05)) ** 2
        disk = np.exp(-r**2 * 1.5) * (0.35 + 0.65 * arm)
        bulge = np.exp(-(r**2) * 12.0)
        val = np.clip(disk + bulge, 0, 1)
        img[..., 0] = val * rng.uniform(180, 220)
        img[..., 1] = val * rng.uniform(170, 210)
        img[..., 2] = val * rng.uniform(150, 200)
        img += rng.normal(0, 6, img.shape)
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def _proc_star_cluster(count: int, image_size: int, rng: np.random.Generator) -> List[Array]:
    """Procedural globular star cluster: dense central swarm of point sources."""
    out = []
    for _ in range(count):
        s = image_size
        img = np.zeros((s, s, 3), dtype=np.float32)
        cy, cx = s / 2, s / 2
        n_stars = rng.integers(180, 420)
        for _ in range(n_stars):
            rad = abs(rng.normal(0, s * 0.16))
            ang = rng.uniform(0, 2 * np.pi)
            x = int(cx + rad * np.cos(ang))
            y = int(cy + rad * np.sin(ang))
            if 0 <= x < s and 0 <= y < s:
                b = rng.uniform(120, 255)
                img[y, x, :] = min(255, b)
                for ddx in (-1, 0, 1):
                    for ddy in (-1, 0, 1):
                        xx, yy = x + ddx, y + ddy
                        if 0 <= xx < s and 0 <= yy < s:
                            img[yy, xx, :] = np.maximum(img[yy, xx, :], b * 0.4)
        img += rng.normal(0, 4, img.shape)
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def _proc_planetary(count: int, image_size: int, rng: np.random.Generator) -> List[Array]:
    """Procedural planetary object: a lit disk with limb shading + rim."""
    out = []
    for _ in range(count):
        s = image_size
        img = np.zeros((s, s, 3), dtype=np.float32)
        cy, cx = s / 2, s / 2
        radius = rng.uniform(s * 0.22, s * 0.38)
        Y, X = np.ogrid[:s, :s]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = dist < radius
        lx, ly = rng.uniform(-1, 1), rng.uniform(-1, 1)
        shade = np.clip(0.5 + 0.5 * ((X - cx) / radius * lx + (Y - cy) / radius * ly), 0, 1)
        base = np.array([rng.uniform(120, 220), rng.uniform(90, 190), rng.uniform(60, 160)])
        for c in range(3):
            ch = np.zeros((s, s), dtype=np.float32)
            ch[mask] = base[c] * shade[mask]
            img[..., c] = ch
        img += rng.normal(0, 3, img.shape)
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def procedural_class(cls: str, count: int, image_size: int, seed: int) -> List[Array]:
    """Dispatch to the right procedural generator, reusing repo patterns where possible."""
    rng = np.random.default_rng(seed)
    if cls == "nebula" and _HAVE_REPO_GENERATORS:
        return [to_rgb_uint8(a, image_size) for a in _repo_nebula(count)]
    if cls == "elliptical_galaxy" and _HAVE_REPO_GENERATORS:
        return [to_rgb_uint8(a, image_size) for a in _repo_elliptical(count)]
    if cls == "spiral_galaxy":
        return _proc_spiral(count, image_size, rng)
    if cls == "elliptical_galaxy":
        # local elliptical fallback if repo generators unavailable
        out = []
        for _ in range(count):
            s = image_size
            img = np.zeros((s, s, 3), dtype=np.float32)
            cy, cx = s / 2, s / 2
            a = rng.uniform(s * 0.14, s * 0.32)
            b = rng.uniform(s * 0.10, s * 0.24)
            Y, X = np.ogrid[:s, :s]
            r = np.sqrt(((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2)
            val = np.exp(-r**2 * 1.5) * rng.uniform(180, 220)
            for c in range(3):
                img[..., c] = val
            img += rng.normal(0, 6, img.shape)
            out.append(np.clip(img, 0, 255).astype(np.uint8))
        return out
    if cls == "nebula":
        out = []
        palette = [(255, 100, 150), (100, 200, 255), (200, 150, 255),
                   (255, 200, 100), (150, 255, 200)]
        for i in range(count):
            s = image_size
            img = np.zeros((s, s, 3), dtype=np.float32)
            cy, cx = s / 2, s / 2
            color = palette[i % len(palette)]
            Y, X = np.ogrid[:s, :s]
            dx, dy = (X - cx) / (s * 0.36), (Y - cy) / (s * 0.36)
            r = np.sqrt(dx**2 + dy**2)
            filament = np.sin(dx * 5) * np.cos(dy * 5) * 0.5
            intensity = np.clip(np.exp(-r**2 * 0.8) * (1 + filament), 0, None)
            for c in range(3):
                img[..., c] = intensity * color[c]
            img += rng.normal(0, 10, img.shape)
            out.append(np.clip(img, 0, 255).astype(np.uint8))
        return out
    if cls == "star_cluster":
        return _proc_star_cluster(count, image_size, rng)
    if cls == "planetary":
        return _proc_planetary(count, image_size, rng)
    raise ValueError(f"No procedural generator for class '{cls}'")


# ---------------------------------------------------------------------------
# Assembly + split + write
# ---------------------------------------------------------------------------

def assemble(per_class: int, image_size: int, seed: int, real_only: bool
             ) -> Tuple[Dict[str, List[Array]], Dict[str, dict]]:
    """Return (images_by_class, manifest_by_class) honouring real-first strategy."""
    py_rng = random.Random(seed)
    images_by_class: Dict[str, List[Array]] = {}
    manifest: Dict[str, dict] = {}

    print("\n[1/3] Acquiring REAL sources...")
    galaxy = get_galaxy10(per_class, image_size, py_rng)
    kaggle = get_kaggle_classes(["nebula", "star_cluster", "planetary"],
                                per_class, image_size)
    real = {**galaxy, **kaggle}

    print("\n[2/3] Filling / balancing each class to "
          f"{per_class} images...")
    for cls in CLASSES:
        arrs: List[Array] = []
        sources: List[str] = []
        if cls in real:
            arrs = list(real[cls][0])[:per_class]
            sources.append(real[cls][1])

        shortfall = per_class - len(arrs)
        if shortfall > 0:
            if real_only:
                if not arrs:
                    raise RuntimeError(
                        f"--real-only set but no real images for '{cls}'. "
                        f"Provide astroNN/Kaggle access or drop --real-only.")
                print(f"  {cls}: real-only, keeping {len(arrs)} (no top-up).")
            else:
                gen = procedural_class(cls, shortfall, image_size, seed + hash(cls) % 9973)
                arrs.extend(gen)
                sources.append("procedural-fallback")
                print(f"  {cls}: {len(arrs) - shortfall} real + {shortfall} procedural.")
        else:
            print(f"  {cls}: {len(arrs)} real (no fallback needed).")

        arrs = dedupe(arrs)[:per_class]
        images_by_class[cls] = arrs
        manifest[cls] = {
            "source": " + ".join(sources) if sources else "procedural-fallback",
            "count": len(arrs),
            "image_size": image_size,
            "is_real": bool(cls in real),
        }
    return images_by_class, manifest


def stratified_split(images_by_class: Dict[str, List[Array]], seed: int
                     ) -> Dict[str, List[Tuple[str, Array]]]:
    """DISJOINT stratified 80/10/10 split. Returns {split: [(class, array), ...]}."""
    from sklearn.model_selection import StratifiedShuffleSplit

    items: List[Tuple[str, Array]] = []
    labels: List[int] = []
    for idx, cls in enumerate(CLASSES):
        for arr in images_by_class[cls]:
            items.append((cls, arr))
            labels.append(idx)
    y = np.array(labels)
    X = np.arange(len(items))

    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, temp_idx = next(s1.split(X, y))
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(s2.split(temp_idx, y[temp_idx]))

    return {
        "train": [items[i] for i in train_idx],
        "val": [items[i] for i in temp_idx[val_rel]],
        "test": [items[i] for i in temp_idx[test_rel]],
    }


def write_splits(split_data: Dict[str, List[Tuple[str, Array]]]
                 ) -> Dict[str, Dict[str, List[str]]]:
    """Write images to disk; return per-split per-class list of MD5 file hashes."""
    hashes: Dict[str, Dict[str, List[str]]] = {}
    for split, entries in split_data.items():
        split_dir = PROCESSED_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir)  # idempotent rebuild
        counters: Dict[str, int] = defaultdict(int)
        hashes[split] = defaultdict(list)
        for cls, arr in entries:
            class_dir = split_dir / cls
            class_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{cls}_{counters[cls]:04d}.png"
            path = class_dir / fname
            Image.fromarray(arr).save(path)
            counters[cls] += 1
            hashes[split][cls].append(md5_bytes(path.read_bytes()))
    return hashes


def assert_no_leakage(hashes: Dict[str, Dict[str, List[str]]]) -> None:
    """Assert no image content-hash appears in more than one split."""
    per_split = {split: set(h for lst in cls_map.values() for h in lst)
                 for split, cls_map in hashes.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    leaks = 0
    for a, b in pairs:
        overlap = per_split[a] & per_split[b]
        if overlap:
            leaks += len(overlap)
            print(f"  LEAK: {len(overlap)} shared hashes between {a} and {b}!")
    assert leaks == 0, f"Data leakage detected across splits ({leaks} shared hashes)"
    total = sum(len(s) for s in per_split.values())
    print(f"  Leakage check PASSED: 0 shared MD5 hashes across "
          f"{total} images ({len(per_split['train'])}/"
          f"{len(per_split['val'])}/{len(per_split['test'])} train/val/test).")


def compute_class_weights(split_data: Dict[str, List[Tuple[str, Array]]]) -> Dict[str, float]:
    """Inverse-frequency weights from the TRAIN split, keyed by exact CLASSES names."""
    counts = {c: 0 for c in CLASSES}
    for cls, _ in split_data["train"]:
        counts[cls] += 1
    total = sum(counts.values())
    n = len(CLASSES)
    weights = {c: (total / (n * cnt) if cnt > 0 else 0.0) for c, cnt in counts.items()}
    max_w = max(weights.values()) or 1.0
    return {c: round(min(w / max_w * 5.0, 5.0), 6) for c, w in weights.items()}


def print_summary(manifest: Dict[str, dict],
                  split_data: Dict[str, List[Tuple[str, Array]]]) -> None:
    counts = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    for split, entries in split_data.items():
        for cls, _ in entries:
            counts[split][cls] += 1
    print("\n" + "=" * 84)
    print(f"{'class':20s} {'source':32s} {'train':>6s} {'val':>5s} {'test':>5s}")
    print("-" * 84)
    for cls in CLASSES:
        src = manifest[cls]["source"]
        src_short = ("procedural-fallback" if src.startswith("procedural")
                     else ("REAL:" + src.split("|")[0].split(":")[0].strip()[:25]))
        print(f"{cls:20s} {src_short:32s} "
              f"{counts['train'][cls]:>6d} {counts['val'][cls]:>5d} {counts['test'][cls]:>5d}")
    print("=" * 84)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assemble data/processed/{train,val,test}/<class>/ (real-first, safe-fallback).")
    p.add_argument("--per-class", type=int, default=500,
                   help="Target images per class (default: 500).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--image-size", type=int, default=None,
                   help="Output square image size (default: config data.image_size or 224).")
    p.add_argument("--real-only", action="store_true",
                   help="Disable procedural fallback; use only real images.")
    return p.parse_args()


def resolve_image_size(cli_value: Optional[int]) -> int:
    if cli_value is not None:
        return cli_value
    try:
        from utils import load_config
        return int(load_config()["data"]["image_size"])
    except Exception:
        return 224


def main() -> None:
    args = parse_args()
    image_size = resolve_image_size(args.image_size)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 84)
    print("Galaxy-X-os -- Data Preparation (REAL-first, SAFE-fallback, idempotent)")
    print(f"per_class={args.per_class} | seed={args.seed} | image_size={image_size} "
          f"| real_only={args.real_only}")
    print("=" * 84)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    images_by_class, manifest = assemble(args.per_class, image_size, args.seed, args.real_only)

    print("\n[3/3] Stratified 80/10/10 split + write + leakage check...")
    split_data = stratified_split(images_by_class, args.seed)
    hashes = write_splits(split_data)
    assert_no_leakage(hashes)

    # class_weights.json (dir already created above)
    weights = compute_class_weights(split_data)
    with open(PROCESSED_DIR / "class_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
    print(f"  Wrote {PROCESSED_DIR / 'class_weights.json'}: {weights}")

    # DATA_MANIFEST.json -- the honesty record
    manifest_doc = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_class_target": args.per_class,
        "image_size": image_size,
        "seed": args.seed,
        "real_only": args.real_only,
        "classes": manifest,
        "split_ratios": SPLITS,
    }
    with open(PROCESSED_DIR / "DATA_MANIFEST.json", "w") as f:
        json.dump(manifest_doc, f, indent=2)
    print(f"  Wrote {PROCESSED_DIR / 'DATA_MANIFEST.json'}")

    print_summary(manifest, split_data)
    n_real = sum(1 for c in CLASSES if manifest[c]["is_real"])
    print(f"\nDone. {n_real}/{len(CLASSES)} classes used a REAL source; "
          f"see DATA_MANIFEST.json for the honest per-class record.")
    print("Next: python src/train.py")


if __name__ == "__main__":
    main()
