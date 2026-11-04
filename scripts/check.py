#!/usr/bin/env python3
import hashlib
from pathlib import Path
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def check_cubin(path, sha, errors):
    if not path.is_file():
        errors.append(f"missing cubin: {path}")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        errors.append(f"SHA-256 mismatch: {path}")


def main():
    """kernels.toml: every module has a source or an upstream record, its source and build files exist,
    its prebuilt cubin (and, given a build directory, its built one) matches its hash, and no cubin under
    prebuilt/ is unpinned. Then every captured set under decode/."""
    kernels = tomllib.loads((ROOT / "kernels.toml").read_text())["kernels"]
    built = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    errors = []
    for name, kernel in kernels.items():
        assert re.fullmatch(r"[0-9a-f]{64}", kernel["sha256"]), name
        assert "source" in kernel or "upstream" in kernel, name
        for field in ("source", "build"):
            if field in kernel:
                assert (ROOT / kernel[field]).is_file(), (name, field)
        if "prebuilt" in kernel:
            check_cubin(ROOT / kernel["prebuilt"], kernel["sha256"], errors)
        if built:
            check_cubin(built / f"{name}.cubin", kernel["sha256"], errors)
    pinned = {kernel["prebuilt"] for kernel in kernels.values() if "prebuilt" in kernel}
    stray = {f"prebuilt/{p.name}" for p in (ROOT / "prebuilt").glob("*.cubin")} - pinned
    errors += [f"{p} is in no kernels.toml entry" for p in sorted(stray)]
    print(f"kernels.toml: {len(kernels)} modules, {len(pinned)} prebuilt, sources and hashes OK")
    for table in sorted((ROOT / "decode").glob("*/kernels.toml")):
        check_capture(table, errors)
    if errors:
        raise SystemExit("\n".join(errors))


def check_capture(table, errors):
    """A captured set: every prebuilt cubin matches its hash, every module is either here or says why not."""
    here = table.parent
    kernels = tomllib.loads(table.read_text())["kernels"]
    pinned = set()
    for name, kernel in kernels.items():
        assert re.fullmatch(r"[0-9a-f]{64}", kernel["sha256"]), name
        assert ("prebuilt" in kernel) != ("not_included" in kernel), name
        assert kernel["symbols"], name
        if "prebuilt" in kernel:
            check_cubin(here / kernel["prebuilt"], kernel["sha256"], errors)
            pinned.add(kernel["prebuilt"])
    stray = {f"prebuilt/{p.name}" for p in (here / "prebuilt").glob("*.cubin")} - pinned
    errors += [f"{here.name}: {p} is in no kernels.toml entry" for p in sorted(stray)]
    print(f"{here.name}: {len(kernels)} modules, {len(pinned)} prebuilt, hashes OK")


if __name__ == "__main__":
    main()
