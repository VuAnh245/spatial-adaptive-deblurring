"""Build GoPro split manifests (by sequence).

    python scripts/build_manifests.py --gopro data/GOPRO_Large [--realblur data/RealBlur]

- gopro_sharp_train.txt / gopro_sharp_val.txt: sharp frames, 2 train sequences held out for val, taken
  from videos with a single clip in train and none in test (no recording shared with train or test)
- gopro_sharp_test.txt: sharp frames of the official test sequences
- gopro_test_pairs.txt: blur<TAB>sharp for the 1,111 official test pairs
- realblur_j_test_pairs.txt: from the official RealBlur_J_test_list.txt (gt first, blur second)
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from spatial_deblur.data.manifests import assert_disjoint_scenes, video_of, write_list


def frames(root: Path, split: str, kind: str) -> list[str]:
    # as_posix: manifests must use "/" on every OS (pair building and scene ids split on it)
    return sorted(p.relative_to(root).as_posix() for p in (root / split).glob(f"*/{kind}/*.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gopro", required=True)
    ap.add_argument("--realblur", default=None)
    ap.add_argument("--val-sequences", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    root = Path(a.gopro)
    train_sharp = frames(root, "train", "sharp")
    te = frames(root, "test", "sharp")
    # Val (QĐ-05 follow-up): sequences whose video has exactly ONE clip in train and no clip in test,
    # so val shares no recording with train or test. The official train/test split itself shares videos
    # (reported as a limitation), which is why only val/train and val/test are asserted below.
    clips_per_video: dict[str, set] = {}
    for p in train_sharp:
        clips_per_video.setdefault(video_of(p, 2), set()).add(Path(p).parts[1])
    test_videos = {video_of(p, 2) for p in te}
    candidates = sorted(next(iter(c)) for v, c in clips_per_video.items() if len(c) == 1 and v not in test_videos)
    if len(candidates) < a.val_sequences:
        raise SystemExit(f"only {len(candidates)} candidate val sequences: {candidates}")
    random.Random(a.seed).shuffle(candidates)
    val_seqs = set(candidates[: a.val_sequences])
    tr = [p for p in train_sharp if Path(p).parts[1] not in val_seqs]
    va = [p for p in train_sharp if Path(p).parts[1] in val_seqs]
    assert_disjoint_scenes(tr, va, depth=2, level="video")
    assert_disjoint_scenes(va, te, depth=2, level="video")
    shared = sorted({video_of(p, 2) for p in tr} & test_videos)
    print(f"val candidates ({len(candidates)}): {sorted(candidates)}")
    print(f"official train/test share {len(shared)} of {len(test_videos)} test videos: {shared}")
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
