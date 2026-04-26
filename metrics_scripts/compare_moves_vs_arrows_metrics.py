import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


MovePair = Tuple[str, str]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_uci_move(move: str) -> Optional[MovePair]:
    move = move.strip().lower()
    if len(move) < 4:
        return None
    src = move[:2]
    dst = move[2:4]
    if is_square(src) and is_square(dst):
        return (src, dst)
    return None


def is_square(s: str) -> bool:
    return len(s) == 2 and s[0] in "abcdefgh" and s[1] in "12345678"


def dedupe_keep_order(items: List[MovePair]) -> List[MovePair]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# -------------------------
# MOVES
# -------------------------

def extract_moves(move_json: dict, use_valid_only: bool = False) -> List[MovePair]:
    """
    Expected move schema:
    {
      "id": "...",
      ...
      "move_data": {
        "candidate_moves": [...],
        "valid_candidate_moves": [...],
        "invalid_candidate_moves": [...]
      }
    }
    """
    move_data = move_json.get("move_data", {})

    raw_moves = []
    if use_valid_only:
        raw_moves = move_data.get("valid_candidate_moves", [])
    else:
        raw_moves = move_data.get("candidate_moves", [])

    parsed = []
    for m in raw_moves:
        if isinstance(m, str):
            p = parse_uci_move(m)
            if p:
                parsed.append(p)

    return dedupe_keep_order(parsed)


# -------------------------
# ANNOTATION ARROWS
# -------------------------

def extract_candidate_arrows(annotation_json: dict, prefer_valid: bool = True) -> List[MovePair]:
    """
    Supports:
    1) model schema
       candidate_move_arrows / valid_candidate_move_arrows
       each arrow has {from, to, piece}

    2) random schema
       candidate_move_arrows
       each arrow has {from, to, piece}

    3) gold schema
       arrows: [{from_square, to_square, kind}]
       keep only kind == "candidate_move"
    """
    pairs = []

    # gold schema
    if "arrows" in annotation_json:
        for arrow in annotation_json.get("arrows", []):
            if not isinstance(arrow, dict):
                continue
            if arrow.get("kind") != "candidate_move":
                continue
            src = str(arrow.get("from_square", "")).lower()
            dst = str(arrow.get("to_square", "")).lower()
            if is_square(src) and is_square(dst):
                pairs.append((src, dst))
        return dedupe_keep_order(pairs)

    # model schema
    if prefer_valid and "valid_candidate_move_arrows" in annotation_json:
        arrow_list = annotation_json.get("valid_candidate_move_arrows", [])
    else:
        arrow_list = annotation_json.get("candidate_move_arrows", [])

    for arrow in arrow_list:
        if not isinstance(arrow, dict):
            continue
        src = str(arrow.get("from", "")).lower()
        dst = str(arrow.get("to", "")).lower()
        if is_square(src) and is_square(dst):
            pairs.append((src, dst))

    return dedupe_keep_order(pairs)


def extract_all_arrows(annotation_json: dict, prefer_valid: bool = True) -> List[MovePair]:
    """
    Optional broader metric:
    compare moves against all arrows, not just candidate move arrows.
    Useful if you want to see whether moves align with any visual grounding at all.

    For model/random:
      candidate_move_arrows + threat_arrows
    For gold:
      all arrows
    """
    pairs = []

    # gold schema
    if "arrows" in annotation_json:
        for arrow in annotation_json.get("arrows", []):
            if not isinstance(arrow, dict):
                continue
            src = str(arrow.get("from_square", "")).lower()
            dst = str(arrow.get("to_square", "")).lower()
            if is_square(src) and is_square(dst):
                pairs.append((src, dst))
        return dedupe_keep_order(pairs)

    candidate_key = "valid_candidate_move_arrows" if prefer_valid and "valid_candidate_move_arrows" in annotation_json else "candidate_move_arrows"
    threat_key = "valid_threat_arrows" if prefer_valid and "valid_threat_arrows" in annotation_json else "threat_arrows"

    for key in [candidate_key, threat_key]:
        for arrow in annotation_json.get(key, []):
            if not isinstance(arrow, dict):
                continue
            src = str(arrow.get("from", "")).lower()
            dst = str(arrow.get("to", "")).lower()
            if is_square(src) and is_square(dst):
                pairs.append((src, dst))

    return dedupe_keep_order(pairs)


# -------------------------
# FILE MATCHING
# -------------------------

def build_json_map(folder: Path) -> Dict[str, Path]:
    out = {}
    for p in folder.rglob("*.json"):
        out[p.stem] = p
    return out


# -------------------------
# METRICS
# -------------------------

def compute_metrics(moves: List[MovePair], arrows: List[MovePair]) -> dict:
    move_set = set(moves)
    arrow_set = set(arrows)
    overlap = move_set & arrow_set

    top1_exact = 1.0 if moves and arrows and moves[0] == arrows[0] else 0.0
    top1_in_any = 1.0 if moves and moves[0] in arrow_set else 0.0
    any_overlap = 1.0 if overlap else 0.0

    move_support = len(overlap) / len(move_set) if move_set else 0.0
    arrow_realization = len(overlap) / len(arrow_set) if arrow_set else 0.0

    return {
        "top1_exact": top1_exact,
        "top1_in_any_arrow": top1_in_any,
        "any_overlap": any_overlap,
        "move_support": move_support,
        "arrow_realization": arrow_realization,
        "num_moves": len(moves),
        "num_arrows": len(arrows),
        "num_overlap": len(overlap),
    }


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(rows: List[dict]) -> dict:
    return {
        "n_compared": len(rows),
        "top1_exact_pct": 100.0 * mean([r["top1_exact"] for r in rows]),
        "top1_in_any_arrow_pct": 100.0 * mean([r["top1_in_any_arrow"] for r in rows]),
        "any_overlap_pct": 100.0 * mean([r["any_overlap"] for r in rows]),
        "move_support_pct": 100.0 * mean([r["move_support"] for r in rows]),
        "arrow_realization_pct": 100.0 * mean([r["arrow_realization"] for r in rows]),
        "avg_num_moves": mean([r["num_moves"] for r in rows]),
        "avg_num_arrows": mean([r["num_arrows"] for r in rows]),
        "avg_num_overlap": mean([r["num_overlap"] for r in rows]),
    }


def compare_dirs(
    moves_dir: Path,
    annotations_dir: Path,
    out_json: Optional[Path],
    use_valid_moves_only: bool,
    use_all_arrows: bool,
    prefer_valid_arrows: bool,
    show_examples: int,
):
    move_map = build_json_map(moves_dir)
    ann_map = build_json_map(annotations_dir)

    common_ids = sorted(set(move_map.keys()) & set(ann_map.keys()))
    if not common_ids:
        print("[error] no overlapping ids by filename stem")
        print(f"moves_dir={moves_dir}")
        print(f"annotations_dir={annotations_dir}")
        return

    rows = []

    for pid in common_ids:
        move_json = load_json(move_map[pid])
        ann_json = load_json(ann_map[pid])

        moves = extract_moves(move_json, use_valid_only=use_valid_moves_only)

        if use_all_arrows:
            arrows = extract_all_arrows(ann_json, prefer_valid=prefer_valid_arrows)
        else:
            arrows = extract_candidate_arrows(ann_json, prefer_valid=prefer_valid_arrows)

        row = {
            "id": pid,
            "moves": [a + b for a, b in moves],
            "arrows": [a + b for a, b in arrows],
            "move_file": str(move_map[pid]),
            "annotation_file": str(ann_map[pid]),
        }
        row.update(compute_metrics(moves, arrows))
        rows.append(row)

    summary = summarize(rows)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:26s} {v:.4f}")
        else:
            print(f"{k:26s} {v}")

    print("\n=== SAMPLE ROWS ===")
    for row in rows[:show_examples]:
        print(f"\nid: {row['id']}")
        print(f"moves:  {row['moves']}")
        print(f"arrows: {row['arrows']}")
        print(f"top1_exact={row['top1_exact']}")
        print(f"top1_in_any_arrow={row['top1_in_any_arrow']}")
        print(f"any_overlap={row['any_overlap']}")
        print(f"move_support={row['move_support']:.3f}")
        print(f"arrow_realization={row['arrow_realization']:.3f}")

    if out_json:
        payload = {
            "config": {
                "moves_dir": str(moves_dir),
                "annotations_dir": str(annotations_dir),
                "use_valid_moves_only": use_valid_moves_only,
                "use_all_arrows": use_all_arrows,
                "prefer_valid_arrows": prefer_valid_arrows,
            },
            "summary": summary,
            "per_file": rows,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[wrote] {out_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moves-dir", required=True)
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--use-valid-moves-only", action="store_true")
    parser.add_argument("--use-all-arrows", action="store_true")
    parser.add_argument("--no-prefer-valid-arrows", action="store_true")
    parser.add_argument("--show-examples", type=int, default=10)
    args = parser.parse_args()

    compare_dirs(
        moves_dir=Path(args.moves_dir),
        annotations_dir=Path(args.annotations_dir),
        out_json=Path(args.out_json) if args.out_json else None,
        use_valid_moves_only=args.use_valid_moves_only,
        use_all_arrows=args.use_all_arrows,
        prefer_valid_arrows=not args.no_prefer_valid_arrows,
        show_examples=args.show_examples,
    )


if __name__ == "__main__":
    main()