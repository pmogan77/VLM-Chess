#!/usr/bin/env python3
"""
Batch evaluation of all 5 move-generation conditions against Stockfish.

Conditions evaluated:
  - model_moves   : VLM-annotated board  (experimental)
  - plain_moves   : plain board image    (vision baseline)
  - gold_moves    : gold-annotated board (oracle upper bound)
  - random_moves  : random annotations  (ablation)
  - text_only     : FEN text only        (text baseline)

Metrics per puzzle, aggregated across all 99 puzzles:
  1. Hallucination rate   : % of generated moves that are illegal
  2. Recall@k             : fraction of Stockfish top-k found in legal candidates
  3. Hit@k                : binary, 1 if any candidate is in Stockfish top-k
  4. Top-1 hit rate       : % of puzzles where Stockfish #1 move is in candidates
  5. Centipawn gap        : Stockfish best score minus best candidate score
  6. MRR                  : mean reciprocal rank of best candidate in Stockfish top-k

Usage:
  python batch_eval.py --engine /opt/homebrew/bin/stockfish --k 5 --depth 12 --out results/eval_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import chess
import chess.engine

MATE_SCORE = 100_000
BASE = Path("puzzles/gemma-4-31b-it")

CONDITIONS: dict[str, Path] = {
    "model_moves":  BASE / "model_moves",
    "plain_moves":  BASE / "plain_moves",
    "gold_moves":   BASE / "gold_moves",
    "random_moves": BASE / "random_moves",
    "text_only":    BASE / "text_only" / "moves",
}


# ---------------------------------------------------------------------------
# Stockfish helpers (adapted from stockfish_candidate_eval.py)
# ---------------------------------------------------------------------------

def make_limit(depth: int | None, time_limit: float | None) -> chess.engine.Limit:
    if depth is not None:
        return chess.engine.Limit(depth=depth)
    return chess.engine.Limit(time=time_limit or 0.2)


def score_to_cp(score_obj: chess.engine.PovScore, turn: bool) -> tuple[int | None, bool, int | None]:
    pov = score_obj.pov(turn)
    return pov.score(mate_score=MATE_SCORE), pov.is_mate(), pov.mate()


def get_stockfish_top_k(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    k: int,
    limit: chess.engine.Limit,
) -> list[dict]:
    infos = engine.analyse(
        board, limit, multipv=k,
        info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
    )
    if isinstance(infos, dict):
        infos = [infos]
    results = []
    for i, info in enumerate(infos, 1):
        pv = info.get("pv", [])
        if not pv:
            continue
        score_cp, is_mate, mate_in = score_to_cp(info["score"], board.turn)
        results.append({"rank": i, "move": pv[0].uci(), "score_cp": score_cp,
                        "is_mate": is_mate, "mate_in": mate_in})
    return results


def score_legal_candidates(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    legal_moves: list[str],
    limit: chess.engine.Limit,
) -> dict[str, int | None]:
    if not legal_moves:
        return {}
    parsed = [chess.Move.from_uci(m) for m in legal_moves]
    infos = engine.analyse(
        board, limit, multipv=len(parsed), root_moves=parsed,
        info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
    )
    if isinstance(infos, dict):
        infos = [infos]
    scores: dict[str, int | None] = {}
    for info in infos:
        pv = info.get("pv", [])
        if pv:
            cp, _, _ = score_to_cp(info["score"], board.turn)
            scores[pv[0].uci()] = cp
    return scores


# ---------------------------------------------------------------------------
# Per-puzzle evaluation
# ---------------------------------------------------------------------------

@dataclass
class PuzzleResult:
    puzzle_id: str
    fen: str
    condition: str
    candidates_raw: list[str]
    candidates_legal: list[str]
    candidates_illegal: list[str]
    hallucination_rate: float          # metric 1
    stockfish_top_k: list[str]
    stockfish_best_score_cp: int | None
    recall_at_k: float                 # metric 2
    hit_at_k: int                      # metric 3
    top1_hit: int                      # metric 4
    cp_gap: int | None                 # metric 5 (engine_best - candidate_best, lower = better candidate)
    best_candidate_score_cp: int | None
    mrr: float                         # metric 6


def eval_puzzle(
    puzzle_id: str,
    fen: str,
    candidates_raw: list[str],
    condition: str,
    engine: chess.engine.SimpleEngine,
    k: int,
    limit: chess.engine.Limit,
) -> PuzzleResult:
    board = chess.Board(fen)

    # Split legal / illegal
    legal, illegal = [], []
    for uci in dict.fromkeys(candidates_raw):  # dedupe, preserve order
        try:
            m = chess.Move.from_uci(uci)
            (legal if m in board.legal_moves else illegal).append(uci)
        except ValueError:
            illegal.append(uci)

    total = len(legal) + len(illegal)
    hallucination_rate = len(illegal) / total if total else 0.0

    # Stockfish top-k
    topk = get_stockfish_top_k(board, engine, k, limit)
    topk_moves = [t["move"] for t in topk]
    topk_set = set(topk_moves)
    engine_best_cp = topk[0]["score_cp"] if topk else None

    # Recall / hit / top-1
    candidate_set = set(legal)
    overlap = candidate_set & topk_set
    recall_at_k = len(overlap) / len(topk_set) if topk_set else 0.0
    hit_at_k = int(bool(overlap))
    top1_hit = int(bool(topk) and topk[0]["move"] in candidate_set)

    # MRR
    rank_map = {m: i + 1 for i, m in enumerate(topk_moves)}
    found_ranks = [rank_map[m] for m in overlap if m in rank_map]
    mrr = 1.0 / min(found_ranks) if found_ranks else 0.0

    # Centipawn gap
    candidate_scores = score_legal_candidates(board, engine, legal, limit)
    best_candidate_cp = max(candidate_scores.values(), default=None)
    if engine_best_cp is not None and best_candidate_cp is not None:
        cp_gap = engine_best_cp - best_candidate_cp
    else:
        cp_gap = None

    return PuzzleResult(
        puzzle_id=puzzle_id,
        fen=fen,
        condition=condition,
        candidates_raw=list(candidates_raw),
        candidates_legal=legal,
        candidates_illegal=illegal,
        hallucination_rate=hallucination_rate,
        stockfish_top_k=topk_moves,
        stockfish_best_score_cp=engine_best_cp,
        recall_at_k=recall_at_k,
        hit_at_k=hit_at_k,
        top1_hit=top1_hit,
        cp_gap=cp_gap,
        best_candidate_score_cp=best_candidate_cp,
        mrr=mrr,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results: list[PuzzleResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    def mean(vals):
        valid = [v for v in vals if v is not None]
        return sum(valid) / len(valid) if valid else None

    return {
        "n_puzzles": n,
        "hallucination_rate_mean":    mean([r.hallucination_rate for r in results]),
        "recall_at_k_mean":           mean([r.recall_at_k for r in results]),
        "hit_at_k_mean":              mean([r.hit_at_k for r in results]),
        "top1_hit_rate":              mean([r.top1_hit for r in results]),
        "cp_gap_mean":                mean([r.cp_gap for r in results]),
        "mrr_mean":                   mean([r.mrr for r in results]),
        "n_puzzles_no_legal_candidates": sum(1 for r in results if not r.candidates_legal),
    }


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_puzzles(folder: Path) -> list[tuple[str, str, list[str]]]:
    """Returns list of (puzzle_id, fen, candidate_moves)."""
    entries = []
    for f in sorted(folder.glob("*.json")):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        fen = d.get("fen", "").strip()
        moves = d.get("move_data", {}).get("candidate_moves", [])
        if not fen or not isinstance(moves, list):
            continue
        entries.append((f.stem, fen, [m for m in moves if isinstance(m, str)]))
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch Stockfish eval across all conditions.")
    parser.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--out", default="results/eval_summary.json")
    parser.add_argument("--per-puzzle-out", default="results/eval_per_puzzle.json")
    args = parser.parse_args()

    limit = make_limit(args.depth, None)
    out_path = Path(args.out)
    per_puzzle_path = Path(args.per_puzzle_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[PuzzleResult]] = {}
    all_per_puzzle: dict[str, list[dict]] = {}

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    try:
        for condition, folder in CONDITIONS.items():
            puzzles = load_puzzles(folder)
            print(f"\n[{condition}] {len(puzzles)} puzzles found in {folder}")
            results = []
            for i, (pid, fen, candidates) in enumerate(puzzles, 1):
                try:
                    r = eval_puzzle(pid, fen, candidates, condition, engine, args.k, limit)
                    results.append(r)
                    if i % 10 == 0:
                        print(f"  {i}/{len(puzzles)} done...", flush=True)
                except Exception as e:
                    print(f"  SKIP {pid}: {e}", file=sys.stderr)
            all_results[condition] = results
            all_per_puzzle[condition] = [asdict(r) for r in results]
            summary = aggregate(results)
            print(f"  hallucination={summary.get('hallucination_rate_mean', 0):.3f}  "
                  f"recall@{args.k}={summary.get('recall_at_k_mean', 0):.3f}  "
                  f"hit@{args.k}={summary.get('hit_at_k_mean', 0):.3f}  "
                  f"top1_hit={summary.get('top1_hit_rate', 0):.3f}  "
                  f"cp_gap={summary.get('cp_gap_mean')!r}  "
                  f"mrr={summary.get('mrr_mean', 0):.3f}")
    finally:
        engine.quit()

    # Write outputs
    summary_out = {
        cond: aggregate(res) for cond, res in all_results.items()
    }
    with open(out_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    with open(per_puzzle_path, "w") as f:
        json.dump(all_per_puzzle, f, indent=2)

    # Print final table
    metrics = ["hallucination_rate_mean", "recall_at_k_mean", "hit_at_k_mean",
               "top1_hit_rate", "cp_gap_mean", "mrr_mean"]
    header = f"{'condition':<22}" + "".join(f"{m[:14]:>16}" for m in metrics)
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")
    for cond, s in summary_out.items():
        row = f"{cond:<22}"
        for m in metrics:
            v = s.get(m)
            row += f"{v:>16.3f}" if v is not None else f"{'N/A':>16}"
        print(row)
    print(f"\nSummary written to {out_path}")
    print(f"Per-puzzle results written to {per_puzzle_path}")


if __name__ == "__main__":
    main()
