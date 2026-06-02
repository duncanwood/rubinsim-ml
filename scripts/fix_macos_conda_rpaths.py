#!/usr/bin/env python3
"""Repair duplicate LC_RPATH load commands in a conda env's dylibs (macOS).

Why: some conda-forge builds of the gfortran/OpenBLAS stack (e.g.
libgfortran5 13.2.0 build _3, libopenblas 0.3.27, libquadmath) ship a duplicate
`@loader_path` rpath (one with a trailing slash). Recent macOS dyld rejects the
dylib outright, which cascades through libopenblas -> libgfortran -> libquadmath
and breaks `import numpy` with:

    Library not loaded: @rpath/libgfortran.5.dylib ... duplicate LC_RPATH '@loader_path'

The conda-native fix (updating those libs to corrected builds) only exists in a
newer libgfortran *version*, which drags in numpy 2 / newer scipy and tends to
break numpy<2-pinned stacks like rubin_sim + LensCalcPy. This script instead
removes the malformed duplicate rpath in place with `install_name_tool`, which
is the minimal, version-preserving repair. Each modified dylib is backed up
alongside as `<name>.bak-rpath`.

Usage:
    python scripts/fix_macos_conda_rpaths.py [LIBDIR] [--dry-run]

LIBDIR defaults to "$CONDA_PREFIX/lib". Idempotent: dylibs with no duplicate
rpath are left untouched. Reverse with: cp <name>.bak-rpath <name>.
"""
import argparse
import glob
import os
import subprocess
import sys


def rpaths_of(lib):
    out = subprocess.run(["otool", "-l", lib], capture_output=True, text=True).stdout
    rps, expect = [], False
    for line in out.splitlines():
        s = line.strip()
        if s == "cmd LC_RPATH":
            expect = True
        elif expect and s.startswith("path "):
            rps.append(s[len("path "):].split(" (offset")[0])
            expect = False
    return rps


def has_duplicates(rps):
    norm = [r.rstrip("/") for r in rps]
    return len(norm) != len(set(norm))


def dedupe(lib, dry_run):
    rps = rpaths_of(lib)
    if not has_duplicates(rps):
        return False
    canon = []
    for r in rps:
        n = r.rstrip("/") or "/"
        if n not in canon:
            canon.append(n)
    print(f"  {os.path.basename(lib)}: {rps} -> {canon}")
    if dry_run:
        return True
    bak = lib + ".bak-rpath"
    if not os.path.exists(bak):
        subprocess.run(["cp", "-p", lib, bak], check=True)
    # delete every existing rpath (one per call), then re-add the unique set
    while rpaths_of(lib):
        subprocess.run(["install_name_tool", "-delete_rpath", rpaths_of(lib)[0], lib],
                       capture_output=True, text=True)
    for r in canon:
        subprocess.run(["install_name_tool", "-add_rpath", r, lib],
                       capture_output=True, text=True)
    return True


def main():
    if sys.platform != "darwin":
        sys.exit("This script is macOS-only (uses otool/install_name_tool).")
    ap = argparse.ArgumentParser()
    ap.add_argument("libdir", nargs="?", default=os.path.join(
        os.environ.get("CONDA_PREFIX", ""), "lib"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not os.path.isdir(args.libdir):
        sys.exit(f"not a directory: {args.libdir} (pass LIBDIR or activate the env)")

    libs = sorted(set(os.path.realpath(p) for p in glob.glob(os.path.join(args.libdir, "*.dylib"))))
    fixed = sum(dedupe(lib, args.dry_run) for lib in libs)
    verb = "would fix" if args.dry_run else "fixed"
    print(f"{verb} {fixed} dylib(s) with duplicate rpaths (scanned {len(libs)} in {args.libdir})")


if __name__ == "__main__":
    main()
