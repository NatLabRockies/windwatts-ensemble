#!/usr/bin/env python3
"""
Compare tiles.txt (all tiles) vs out_wtkled (produced tiles) and write missing_tiles.txt.

- Accepts tiles printed as packed ints (e.g., 4294967299) or "tx,ty" pairs (e.g., "12,7").
- Detects outputs named like: tile_<ID>.parquet / tile_<ID>.pq / tile_<ID>.csv

Usage:
  python find_missing_tiles.py --tiles tiles.txt --out-dir out_wtkled --out missing_tiles.txt
"""

from __future__ import annotations
import argparse, re
from pathlib import Path

def encode_tile(tx:int, ty:int) -> int:
    return (int(tx) << 32) | (int(ty) & 0xffffffff)

def parse_tile_line(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    if "," in s:
        a, b = s.split(",", 1)
        return encode_tile(int(a.strip()), int(b.strip()))
    return int(s, 10)

def read_all_tiles(tiles_path: Path) -> set[int]:
    ids: set[int] = set()
    with tiles_path.open() as f:
        for line in f:
            t = parse_tile_line(line)
            if t is not None:
                ids.add(t)
    return ids

def find_present_ids(out_dir: Path) -> set[int]:
    rx = re.compile(r"^tile_(\d+)\.(?:parquet|pq|csv)$", re.IGNORECASE)
    present: set[int] = set()
    for fp in out_dir.iterdir():
        if not fp.is_file():
            continue
        m = rx.match(fp.name)
        if m:
            present.add(int(m.group(1), 10))
    return present

def main():
    ap = argparse.ArgumentParser(description="Find missing tile outputs.")
    ap.add_argument("--tiles", type=Path, default="tiles.txt", help="Path to tiles.txt (one tile per line).")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory containing tile_<ID>.* outputs.")
    ap.add_argument("--out", type=Path, default=Path("missing_wtkled_tiles_v2.txt"), help="Where to write missing tile IDs.")
    args = ap.parse_args()

    need = read_all_tiles(args.tiles)
    have = find_present_ids(args.out_dir)

    missing = sorted(need - have)
    args.out.write_text("\n".join(str(t) for t in missing) + ("\n" if missing else ""))
    print(f"[INFO] tiles needed: {len(need)}  present: {len(have)}  missing: {len(missing)}")
    if missing:
        print(f"[INFO] wrote {len(missing)} IDs → {args.out}")

if __name__ == "__main__":
    main()

