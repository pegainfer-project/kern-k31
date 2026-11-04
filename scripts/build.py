#!/usr/bin/env python3
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import shutil
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prebuilt-only", action="store_true", help="copy bundled cubins without invoking nvcc")
    args = parser.parse_args()
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    out = ROOT / "build"
    out.mkdir(exist_ok=True)
    mismatches = []
    for name, kernel in kernels.items():
        dst = out / f"{name}.cubin"
        if "prebuilt" in kernel:
            shutil.copyfile(ROOT / kernel["prebuilt"], dst)
        elif args.prebuilt_only:
            continue
        else:
            defines = [f"-D{k}={v}" for k, v in sorted(kernel.get("defines", {}).items())]
            subprocess.run([os.environ.get("NVCC", "nvcc"), "-cubin", "-arch=sm_103a",
                            *defines, "-o", str(dst), kernel["source"]], cwd=ROOT, check=True)
        sha = hashlib.sha256(dst.read_bytes()).hexdigest()
        matched = sha == kernel["sha256"]
        print(f"{name}: {sha} ({'matches manifest' if matched else 'different from manifest'})", flush=True)
        if not matched:
            mismatches.append(name)
    if mismatches:
        raise SystemExit("Built cubins differ from the pinned manifest: " + ", ".join(mismatches))


if __name__ == "__main__":
    main()
