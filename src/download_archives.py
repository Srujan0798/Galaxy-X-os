#!/usr/bin/env python3
"""
Galaxy-X-os -- src/download_archives.py
=======================================

REAL telescope imagery for the non-galaxy classes (nebula / star_cluster /
planetary) from the **NASA Image Library** -- a public, no-auth, no-API-key
JSON API backed by Hubble / Spitzer / JPL mission imagery.

    Search:  GET https://images-api.nasa.gov/search
             ?q=<QUERY>&media_type=image&page=<N>&page_size=100
    Assets:  direct JPEGs at https://images-assets.nasa.gov/.../<id>~medium.jpg

This module is both:
  * importable  -- ``fetch_nasa_images`` / ``fetch_archive_class`` used by
    ``src/prepare_data.py`` as REAL SOURCE #2 for nebula/star_cluster/planetary.
  * a CLI       -- ``python src/download_archives.py --class nebula --limit 5``
    downloads real JPEGs and prints their titles + sources (a proof harness).

Semantics matter. Per class we run several queries and then apply
include/exclude *purity* filters against the concatenated lowercase
title+description+keywords so that, e.g., a "planetary nebula" is counted as a
nebula and NEVER as a planet, and a "galaxy cluster" is never a star cluster.

No API key. Robust: per-request timeouts, skips failed downloads, MD5 dedupe.
"""

from __future__ import annotations

import io
import ssl
import json
import time
import hashlib
import argparse
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

Array = np.ndarray

NASA_SEARCH = "https://images-api.nasa.gov/search"
NASA_PORTAL = "https://images.nasa.gov"
NASA_API = "https://images-api.nasa.gov"
SOURCE_LABEL = f"NASA Image Library | {NASA_PORTAL} ({NASA_API})"
_UA = {"User-Agent": "galaxy-x-os/1.0 (+data-prep)"}


# ---------------------------------------------------------------------------
# SSL context -- prefer certifi's CA bundle (Python 3.14 ships no system CAs
# on macOS); fall back to the default context if certifi is unavailable.
# ---------------------------------------------------------------------------

def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_SSL = _ssl_context()


# ---------------------------------------------------------------------------
# Image helpers (mirrors src/prepare_data.py to keep outputs identical)
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


def _md5_array(arr: Array) -> str:
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def dedupe(arrs: List[Array]) -> List[Array]:
    """Drop exact-duplicate images (keeps split disjointness honest)."""
    seen, out = set(), []
    for a in arrs:
        h = _md5_array(a)
        if h not in seen:
            seen.add(h)
            out.append(a)
    return out


# ---------------------------------------------------------------------------
# Per-class search queries + purity filters
# ---------------------------------------------------------------------------
# include: item accepted if ANY substring is present in title+desc+keywords.
# exclude: item REJECTED if ANY substring is present (checked first).

CLASS_QUERIES: Dict[str, Dict[str, object]] = {
    "nebula": {
        "queries": [
            "emission nebula", "planetary nebula", "supernova remnant nebula",
            "reflection nebula", "nebula hubble",
        ],
        # A "planetary nebula" IS a nebula -- include it here.
        "include": ["nebula", "supernova remnant"],
        "exclude": [],
    },
    "star_cluster": {
        "queries": [
            "globular star cluster", "open star cluster hubble",
            "star cluster hubble", "globular cluster",
        ],
        "include": ["cluster"],
        # A "galaxy cluster" is galaxies, NOT a star cluster.
        "exclude": ["galaxy cluster", "cluster of galaxies"],
    },
    "planetary": {
        "queries": [
            "planet jupiter", "planet saturn", "planet mars surface",
            "planet neptune", "planet venus", "planet uranus",
            "moon titan", "saturn rings",
        ],
        "include": [
            "jupiter", "saturn", "mars", "neptune", "venus", "uranus",
            "mercury", "pluto", "titan", "moon", "rings",
        ],
        # Never let a "planetary nebula" or a galaxy contaminate this class.
        "exclude": ["nebula", "galaxy"],
    },
}


def _item_text(item: dict) -> str:
    """Concatenated lowercase title + description + keywords for one search item."""
    data = (item.get("data") or [{}])[0]
    title = data.get("title", "") or ""
    desc = data.get("description", "") or ""
    kws = data.get("keywords", []) or []
    return " ".join([title, desc, " ".join(kws)]).lower()


def _passes_filters(text: str, include: List[str], exclude: List[str]) -> bool:
    if any(x in text for x in exclude):
        return False
    return any(inc in text for inc in include) if include else True


def _best_asset_url(links: List[dict]) -> Optional[str]:
    """Pick the direct JPEG, preferring ~medium then ~small/~orig/~thumb."""
    hrefs = [lnk.get("href", "") for lnk in (links or []) if lnk.get("href")]
    for pref in ("~medium.jpg", "~small.jpg", "~orig.jpg", "~thumb.jpg"):
        for h in hrefs:
            if h.lower().endswith(pref):
                return h
    # Fall back to the first image-looking href.
    for h in hrefs:
        if h.lower().endswith((".jpg", ".jpeg", ".png")):
            return h
    return None


def _http_json(url: str, timeout: int = 20) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"  [nasa] search failed for {url!r}: {e}")
        return None


def _http_bytes(url: str, timeout: int = 20) -> Optional[bytes]:
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
            return r.read()
    except Exception as e:
        print(f"  [nasa] download failed for {url!r}: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_nasa_images(query: str, limit: int, image_size: int,
                      include: Optional[List[str]] = None,
                      exclude: Optional[List[str]] = None,
                      max_pages: int = 5, timeout: int = 20,
                      _titles: Optional[List[str]] = None) -> List[Array]:
    """Search the NASA Image Library for ``query`` and return up to ``limit``
    real RGB uint8 images (resized to ``image_size``) that pass the purity
    filters. Paginates until ``limit`` reached or pages run out.

    ``_titles`` (optional) is appended-to with the accepted image titles so the
    caller/CLI can print human-readable provenance.
    """
    include = include or []
    exclude = exclude or []
    out: List[Array] = []
    for page in range(1, max_pages + 1):
        if len(out) >= limit:
            break
        qs = urllib.parse.urlencode({
            "q": query, "media_type": "image", "page": page, "page_size": 100,
        })
        doc = _http_json(f"{NASA_SEARCH}?{qs}", timeout=timeout)
        if not doc:
            break
        items = (doc.get("collection", {}) or {}).get("items", []) or []
        if not items:
            break  # ran out of pages
        for item in items:
            if len(out) >= limit:
                break
            if not _passes_filters(_item_text(item), include, exclude):
                continue
            url = _best_asset_url(item.get("links", []))
            if not url:
                continue
            raw = _http_bytes(url, timeout=timeout)
            if not raw:
                continue
            try:
                arr = to_rgb_uint8(
                    np.array(Image.open(io.BytesIO(raw)).convert("RGB")),
                    image_size,
                )
            except Exception:
                continue
            out.append(arr)
            if _titles is not None:
                title = (item.get("data") or [{}])[0].get("title", "(untitled)")
                _titles.append(title)
    return out


def fetch_archive_class(cls: str, per_class: int, image_size: int,
                        timeout: int = 20,
                        _titles: Optional[List[str]] = None
                        ) -> Tuple[List[Array], str]:
    """Collect up to ``per_class`` real, purity-filtered, deduped images for one
    class from the NASA Image Library. Returns (images, source_label).

    ``cls`` must be one of {"nebula", "star_cluster", "planetary"}.
    """
    if cls not in CLASS_QUERIES:
        return [], SOURCE_LABEL
    spec = CLASS_QUERIES[cls]
    queries: List[str] = spec["queries"]          # type: ignore[assignment]
    include: List[str] = spec["include"]          # type: ignore[assignment]
    exclude: List[str] = spec["exclude"]          # type: ignore[assignment]

    collected: List[Array] = []
    for q in queries:
        if len(collected) >= per_class:
            break
        need = per_class - len(collected)
        got = fetch_nasa_images(q, need, image_size, include, exclude,
                                timeout=timeout, _titles=_titles)
        collected.extend(got)
        collected = dedupe(collected)  # de-dup across queries as we go
        print(f"  [nasa] {cls} <- query {q!r}: +{len(got)} "
              f"(total {len(collected)}/{per_class}).")
    return dedupe(collected)[:per_class], SOURCE_LABEL


# ---------------------------------------------------------------------------
# CLI (proof harness)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download REAL telescope imagery per class from the NASA "
                    "Image Library (no API key).")
    p.add_argument("--class", dest="cls", required=True,
                   choices=list(CLASS_QUERIES.keys()),
                   help="Which class to fetch (nebula/star_cluster/planetary).")
    p.add_argument("--limit", type=int, default=5,
                   help="How many real images to fetch (default: 5).")
    p.add_argument("--image-size", type=int, default=224,
                   help="Output square image size (default: 224).")
    p.add_argument("--save-dir", type=str, default=None,
                   help="Optional dir to save the downloaded JPEGs (proof).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t0 = time.time()
    titles: List[str] = []
    print(f"Fetching {args.limit} REAL '{args.cls}' images from the NASA "
          f"Image Library ({NASA_PORTAL})...")
    arrs, source = fetch_archive_class(args.cls, args.limit, args.image_size,
                                       _titles=titles)
    print("=" * 74)
    print(f"class={args.cls}  downloaded={len(arrs)}  source={source}")
    for i, title in enumerate(titles[:len(arrs)]):
        print(f"  [{i:02d}] {title}")
    print(f"elapsed={time.time() - t0:.1f}s")

    if args.save_dir and arrs:
        from pathlib import Path
        d = Path(args.save_dir)
        d.mkdir(parents=True, exist_ok=True)
        for i, a in enumerate(arrs):
            Image.fromarray(a).save(d / f"{args.cls}_{i:02d}.png")
        print(f"Saved {len(arrs)} PNGs to {d}")


if __name__ == "__main__":
    main()
