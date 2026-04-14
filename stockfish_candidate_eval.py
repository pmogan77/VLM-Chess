"""
Evaluate a list of candidate moves against Stockfish top-k moves for a position.

Example:
python stockfish_candidate_eval.py \
  --fen "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24" \
  --candidates f2g3 b2b1 h8g8 \
  --engine /usr/bin/stockfish \
  --k 5 \
  --depth 18
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Any, Iterable

import chess
import chess.engine

MATE_SCORE = 100000


@dataclass
class EngineMove:
    rank: int
    move_uci: str
    pv: list[str]
    score_cp: int | None
    is_mate: bool
    mate_in: int | None


@dataclass
class CandidateMetrics:
    num_candidates_input: int
    num_candidates_unique: int
    num_candidates_legal: int
    num_candidates_illegal: int
    illegal_candidates: list[str]
    k: int
    overlap_count: int
    overlap_moves: list[str]
    hit_at_k: int
    recall_at_k: float
    precision_at_k: float
    jaccard: float
    best_rank_found: int | None
    mrr: float
    oracle_top1_hit: int


@dataclass
class EvalGap:
    best_engine_score_cp: int | None
    best_candidate_score_cp: int | None
    cp_gap: int | None


@dataclass
class CandidateScore:
    move_uci: str
    score_cp: int | None
    is_mate: bool
    mate_in: int | None



def make_limit(depth: int | None, time_limit: float | None) -> chess.engine.Limit:
    if depth is None and time_limit is None:
        time_limit = 0.2
    if depth is not None:
        return chess.engine.Limit(depth=depth)
    return chess.engine.Limit(time=time_limit)



def score_to_cp(score_obj: chess.engine.PovScore, board_turn: bool) -> tuple[int | None, bool, int | None]:
    pov = score_obj.pov(board_turn)
    return (
        pov.score(mate_score=MATE_SCORE),
        pov.is_mate(),
        pov.mate(),
    )



def get_stockfish_top_k(
    fen: str,
    engine_path: str,
    k: int = 5,
    depth: int | None = 18,
    time_limit: float | None = None,
    threads: int | None = None,
    hash_mb: int | None = None,
) -> list[EngineMove]:
    board = chess.Board(fen)
    limit = make_limit(depth, time_limit)

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        options: dict[str, Any] = {}
        if threads is not None:
            options["Threads"] = threads
        if hash_mb is not None:
            options["Hash"] = hash_mb
        if options:
            engine.configure(options)

        infos = engine.analyse(
            board,
            limit,
            multipv=k,
            info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
        )
    finally:
        engine.quit()

    if isinstance(infos, dict):
        infos = [infos]

    results: list[EngineMove] = []
    for i, info in enumerate(infos, start=1):
        pv = info.get("pv", [])
        if not pv:
            continue
        move = pv[0]
        score_cp, is_mate, mate_in = score_to_cp(info["score"], board.turn)
        results.append(
            EngineMove(
                rank=i,
                move_uci=move.uci(),
                pv=[m.uci() for m in pv],
                score_cp=score_cp,
                is_mate=is_mate,
                mate_in=mate_in,
            )
        )
    return results



def split_candidates_by_legality(
    board: chess.Board,
    candidate_moves: Iterable[str],
) -> tuple[list[str], list[str], list[str]]:
    unique_input = list(dict.fromkeys(candidate_moves))
    legal: list[str] = []
    illegal: list[str] = []

    for uci in unique_input:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            illegal.append(uci)
            continue

        if move in board.legal_moves:
            legal.append(uci)
        else:
            illegal.append(uci)

    return unique_input, legal, illegal



def compare_candidates_to_topk(
    fen: str,
    candidate_moves: Iterable[str],
    stockfish_topk: list[EngineMove],
) -> CandidateMetrics:
    board = chess.Board(fen)
    unique_input, legal_candidates, illegal_candidates = split_candidates_by_legality(board, candidate_moves)

    candidate_set = set(legal_candidates)
    topk_list = [x.move_uci for x in stockfish_topk]
    topk_set = set(topk_list)
    rank_map = {move: i + 1 for i, move in enumerate(topk_list)}

    overlap = candidate_set & topk_set
    found_ranks = sorted(rank_map[m] for m in overlap)
    best_rank_found = min(found_ranks) if found_ranks else None

    hit_at_k = int(len(overlap) > 0)
    recall_at_k = len(overlap) / len(topk_set) if topk_set else 0.0
    precision_at_k = len(overlap) / len(candidate_set) if candidate_set else 0.0
    union = candidate_set | topk_set
    jaccard = len(overlap) / len(union) if union else 0.0
    mrr = (1.0 / best_rank_found) if best_rank_found is not None else 0.0
    oracle_top1_hit = int(stockfish_topk[0].move_uci in candidate_set) if stockfish_topk else 0

    return CandidateMetrics(
        num_candidates_input=len(list(candidate_moves)) if not isinstance(candidate_moves, list) else len(candidate_moves),
        num_candidates_unique=len(unique_input),
        num_candidates_legal=len(candidate_set),
        num_candidates_illegal=len(illegal_candidates),
        illegal_candidates=illegal_candidates,
        k=len(topk_set),
        overlap_count=len(overlap),
        overlap_moves=sorted(overlap, key=lambda m: rank_map[m]),
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        jaccard=jaccard,
        best_rank_found=best_rank_found,
        mrr=mrr,
        oracle_top1_hit=oracle_top1_hit,
    )



def score_candidate_moves(
    fen: str,
    engine_path: str,
    candidate_moves: Iterable[str],
    depth: int | None = 18,
    time_limit: float | None = None,
    threads: int | None = None,
    hash_mb: int | None = None,
) -> list[CandidateScore]:
    board = chess.Board(fen)
    _, legal_candidates, _ = split_candidates_by_legality(board, candidate_moves)

    parsed_moves: list[chess.Move] = [chess.Move.from_uci(uci) for uci in legal_candidates]
    if not parsed_moves:
        return []

    limit = make_limit(depth, time_limit)

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        options: dict[str, Any] = {}
        if threads is not None:
            options["Threads"] = threads
        if hash_mb is not None:
            options["Hash"] = hash_mb
        if options:
            engine.configure(options)

        infos = engine.analyse(
            board,
            limit,
            multipv=len(parsed_moves),
            root_moves=parsed_moves,
            info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
        )
    finally:
        engine.quit()

    if isinstance(infos, dict):
        infos = [infos]

    scored: list[CandidateScore] = []
    for info in infos:
        pv = info.get("pv", [])
        if not pv:
            continue
        move = pv[0]
        score_cp, is_mate, mate_in = score_to_cp(info["score"], board.turn)
        scored.append(
            CandidateScore(
                move_uci=move.uci(),
                score_cp=score_cp,
                is_mate=is_mate,
                mate_in=mate_in,
            )
        )

    scored.sort(
        key=lambda x: (-10**12 if x.score_cp is None else -x.score_cp, x.move_uci)
    )
    return scored



def eval_gap_to_best(
    stockfish_topk: list[EngineMove],
    scored_candidates: list[CandidateScore],
) -> EvalGap:
    if not stockfish_topk or not scored_candidates:
        return EvalGap(
            best_engine_score_cp=None,
            best_candidate_score_cp=None,
            cp_gap=None,
        )

    best_engine = stockfish_topk[0].score_cp
    best_candidate = scored_candidates[0].score_cp
    if best_engine is None or best_candidate is None:
        cp_gap = None
    else:
        cp_gap = best_engine - best_candidate

    return EvalGap(
        best_engine_score_cp=best_engine,
        best_candidate_score_cp=best_candidate,
        cp_gap=cp_gap,
    )



def format_summary(
    fen: str,
    topk: list[EngineMove],
    metrics: CandidateMetrics,
    candidate_scores: list[CandidateScore],
    gap: EvalGap,
) -> str:
    lines: list[str] = []
    lines.append("FEN:")
    lines.append(fen)
    lines.append("")
    lines.append("Stockfish top-k:")
    for row in topk:
        mate_text = f", mate_in={row.mate_in}" if row.is_mate else ""
        lines.append(
            f"  rank={row.rank} move={row.move_uci} score_cp={row.score_cp}{mate_text} pv={' '.join(row.pv)}"
        )
    lines.append("")
    lines.append("Metrics:")
    for key, value in asdict(metrics).items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Candidate scores:")
    if candidate_scores:
        for row in candidate_scores:
            mate_text = f", mate_in={row.mate_in}" if row.is_mate else ""
            lines.append(f"  move={row.move_uci} score_cp={row.score_cp}{mate_text}")
    else:
        lines.append("  <no legal candidate moves>")
    lines.append("")
    lines.append("Gap to best:")
    for key, value in asdict(gap).items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare candidate moves against Stockfish top-k for a FEN."
    )
    parser.add_argument("--fen", required=True, help="Position FEN.")
    parser.add_argument(
        "--candidates",
        nargs="+",
        required=True,
        help="Candidate moves in UCI, e.g. e2e4 g1f3 d2d4",
    )
    parser.add_argument("--engine", required=True, help="Path to Stockfish executable.")
    parser.add_argument("--k", type=int, default=5, help="Top-k Stockfish moves to retrieve.")
    parser.add_argument("--depth", type=int, default=18, help="Engine search depth.")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Optional engine time limit in seconds. Ignored if depth is set.",
    )
    parser.add_argument("--threads", type=int, default=None, help="Optional Stockfish Threads option.")
    parser.add_argument("--hash-mb", type=int, default=None, help="Optional Stockfish Hash size in MB.")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write full results as JSON.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    topk = get_stockfish_top_k(
        fen=args.fen,
        engine_path=args.engine,
        k=args.k,
        depth=args.depth,
        time_limit=args.time_limit,
        threads=args.threads,
        hash_mb=args.hash_mb,
    )
    metrics = compare_candidates_to_topk(args.fen, args.candidates, topk)
    candidate_scores = score_candidate_moves(
        fen=args.fen,
        engine_path=args.engine,
        candidate_moves=args.candidates,
        depth=args.depth,
        time_limit=args.time_limit,
        threads=args.threads,
        hash_mb=args.hash_mb,
    )
    gap = eval_gap_to_best(topk, candidate_scores)

    print(format_summary(args.fen, topk, metrics, candidate_scores, gap))

    if args.json_out:
        payload = {
            "fen": args.fen,
            "candidate_moves": args.candidates,
            "stockfish_topk": [asdict(x) for x in topk],
            "metrics": asdict(metrics),
            "candidate_scores": [asdict(x) for x in candidate_scores],
            "gap_to_best": asdict(gap),
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
