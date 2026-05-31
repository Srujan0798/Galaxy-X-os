#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- Final Packaging Script

Creates the submission zip with all required files.
"""

import zipfile
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ZIP = PROJECT_ROOT / "Galaxy-X-os-Submission.zip"

REQUIRED_DIRS = [
    "src",
    "app",
    "notebooks",
    "config",
    "checkpoints",
    "results",
    "docs",
]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "LICENSE",
    ".gitignore",
]


def verify_files():
    """Verify all required files exist."""
    print("=" * 60)
    print("SCALE x ODYSSEY -- Submission Verification")
    print("=" * 60)

    missing = []

    for d in REQUIRED_DIRS:
        path = PROJECT_ROOT / d
        if not path.exists():
            missing.append(f"Directory: {d}")
            print(f"  ❌ Missing: {d}/")
        else:
            print(f"  ✅ {d}/")

    for f in REQUIRED_FILES:
        path = PROJECT_ROOT / f
        if not path.exists():
            missing.append(f"File: {f}")
            print(f"  ❌ Missing: {f}")
        else:
            print(f"  ✅ {f}")

    if missing:
        print(f"\n❌ {len(missing)} items missing!")
        return False
    else:
        print("\n✅ All required files present!")
        return True


def create_zip():
    """Create submission zip."""
    print("\n" + "=" * 60)
    print("Creating submission zip...")
    print("=" * 60)

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            root_path = Path(root)

            rel_root = root_path.relative_to(PROJECT_ROOT)

            if rel_root.name in ['.git', '__pycache__', '.pytest_cache', 'node_modules']:
                continue

            for file in files:
                if file.endswith('.pyc') or file.startswith('.'):
                    continue

                file_path = root_path / file
                arcname = str(file_path.relative_to(PROJECT_ROOT))
                zf.write(file_path, arcname)
                print(f"  + {arcname}")

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\n✅ Created: {OUTPUT_ZIP} ({size_mb:.1f} MB)")


def main():
    if verify_files():
        create_zip()
        print("\n" + "=" * 60)
        print("🎉 SUBMISSION READY!")
        print("=" * 60)
    else:
        print("\n❌ Cannot create zip - missing files!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
