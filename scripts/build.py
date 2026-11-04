#!/usr/bin/env python3
"""Build every module kernels.toml names into build/<name>.cubin and check its sha256.

A module is either compiled from `source` with its `flags` and `includes` sets (none by default)
and `defines`, or copied from `prebuilt`. `--update` pins the hashes of this build instead of
failing on a change; `--prebuilt-only` copies the bundled cubins without nvcc.
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOML = ROOT / "kernels.toml"


def build(name, k, cfg, nvcc):
    dst = ROOT / "build" / f"{name}.cubin"
    if "prebuilt" in k:
        shutil.copyfile(ROOT / k["prebuilt"], dst)
        return dst
    flags = cfg["flags"][k["flags"]] if "flags" in k else []
    incs = [f"-I{i}" for s in k.get("includes", []) for i in cfg["includes"][s]]
    defs = [f"-D{a}={b}" for a, b in sorted(k.get("defines", {}).items())]
    subprocess.run([nvcc, "-cubin", f"-arch={cfg['toolchain']['arch']}", *flags, *incs, *defs, "-o", str(dst),
                    k["source"]], cwd=ROOT, check=True)
    return dst


def pin(text, name, sha):
    head = f'[kernels."{name}"]\n'
    i = text.index(head)
    j = text.find("\n[", i + len(head))
    j = len(text) if j < 0 else j + 1
    block = re.sub(r'sha256 = "[0-9a-f]*"\n', "", text[i:j])
    return text[:i] + block.replace(head, f'{head}sha256 = "{sha}"\n') + text[j:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prebuilt-only", action="store_true", help="copy bundled cubins without invoking nvcc")
    ap.add_argument("--update", action="store_true", help="pin the hashes of this build")
    ap.add_argument("names", nargs="*", help="modules to build (default: all)")
    a = ap.parse_args()
    cfg = tomllib.loads(TOML.read_text())
    nvcc = os.environ.get("NVCC", "nvcc")
    if not a.prebuilt_only:
        ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout
        want = cfg["toolchain"]["nvcc"]
        if f"V{want}" not in ver:
            raise SystemExit(f"nvcc {want} expected (kernels.toml), got: {ver.strip().splitlines()[-1]}")
    (ROOT / "build").mkdir(exist_ok=True)
    text, bad = TOML.read_text(), []
    for name, k in cfg["kernels"].items():
        if (a.names and name not in a.names) or (a.prebuilt_only and "prebuilt" not in k):
            continue
        sha = hashlib.sha256(build(name, k, cfg, nvcc).read_bytes()).hexdigest()
        ok = sha == k.get("sha256")
        print(f"{name}: {sha} ({'matches' if ok else 'differs from'} kernels.toml)", flush=True)
        if not ok and a.update:
            text = pin(text, name, sha)
        elif not ok:
            bad.append(name)
    if a.update:
        TOML.write_text(text)
    if bad:
        raise SystemExit("built cubins differ from kernels.toml: " + ", ".join(bad))


if __name__ == "__main__":
    main()
