"""Build GoPro split manifests (by sequence).

    python scripts/build_manifests.py --gopro data/GOPRO_Large [--realblur data/RealBlur]

- gopro_sharp_train.txt / gopro_sharp_val.txt: sharp frames, 2 train sequences held out for val
- gopro_sharp_test.txt: sharp frames of the official test sequences
- gopro_test_pairs.txt: blur<TAB>sharp for the 1,111 official test pairs
- realblur_j_test_pairs.txt: from the official RealBlur_J_test_list.txt (gt first, blur second)
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from spatial_deblur.data.manifests import assert_disjoint_scenes, write_list


def frames(root: Path, split: str, kind: str) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in (root / split).glob(f"*/{kind}/*.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gopro", required=True)
    ap.add_argument("--realblur", default=None)
    ap.add_argument("--val-sequences", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    root = Path(a.gopro)
    train_sharp = frames(root, "train", "sharp")
    seqs = sorted({Path(p).parts[1] for p in train_sharp})
    random.Random(a.seed).shuffle(seqs)
    val_seqs = set(seqs[: a.val_sequences])
    tr = [p for p in train_sharp if Path(p).parts[1] not in val_seqs]
    va = [p for p in train_sharp if Path(p).parts[1] in val_seqs]
    te = frames(root, "test", "sharp")
    assert_disjoint_scenes(tr, va, te, depth=2)
    write_list("manifests/gopro_sharp_train.txt", tr)
    write_list("manifests/gopro_sharp_val.txt", va)
    write_list("manifests/gopro_sharp_test.txt", te)
    pairs = [f"{p.replace('/sharp/', '/blur/')}\t{p}" for p in te]
    write_list("manifests/gopro_test_pairs.txt", pairs)
    print(f"train {len(tr)}  val {len(va)} ({sorted(val_seqs)})  test {len(te)}")
    if a.realblur:
        lst = Path(a.realblur) / "RealBlur_J_test_list.txt"
        rows = [ln.split() for ln in lst.read_text().splitlines() if ln.strip()]
        write_list("manifests/realblur_j_test_pairs.txt", [f"{b}\t{g}" for g, b in rows])
        print(f"realblur-J test {len(rows)}")


if __name__ == "__main__":
    main()
