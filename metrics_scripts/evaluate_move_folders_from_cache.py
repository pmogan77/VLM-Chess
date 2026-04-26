#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

import chess
import chess.engine

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


MATE_SCORE = 100000

# FOLDER_MAP = {
#     "gold_moves": "gold_moves",
#     "model_moves": "model_moves",
#     "plain_moves": "plain_moves",
#     "random_moves": "random_moves",
#     "text_moves": "text_only/moves",
# }

FOLDER_MAP = {
    "gold_moves": "gold_moves",
    "gold_moves_fen": "gold_moves_fen",

    "plain_moves": "plain_moves",
    "plain_moves_fen": "plain_moves_fen",

    "random_moves": "random_moves",
    "random_moves_fen": "random_moves_fen",

    "model_moves": "model_moves",
    "model_moves_fen": "model_moves_fen",

    "text_only": "text_only",
}


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
class CandidateScore:
    move_uci: str
    score_cp: int | None
    is_mate: bool
    mate_in: int | None


@dataclass
class EvalGap:
    best_engine_score_cp: int | None
    best_candidate_score_cp: int | None
    cp_gap: int | None


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


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


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
    stockfish_topk: list[dict[str, Any]],
) -> CandidateMetrics:
    board = chess.Board(fen)
    unique_input, legal_candidates, illegal_candidates = split_candidates_by_legality(board, candidate_moves)

    candidate_set = set(legal_candidates)
    topk_list = [x["move_uci"] for x in stockfish_topk]
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
    oracle_top1_hit = int(topk_list[0] in candidate_set) if topk_list else 0

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


def score_candidate_moves_for_fen(
    engine_path: str,
    fen: str,
    candidate_moves: Iterable[str],
    depth: int | None,
    time_limit: float | None,
    threads: int | None,
    hash_mb: int | None,
) -> list[CandidateScore]:
    board = chess.Board(fen)
    _, legal_candidates, _ = split_candidates_by_legality(board, candidate_moves)

    parsed_moves: list[chess.Move] = [chess.Move.from_uci(uci) for uci in legal_candidates]
    if not parsed_moves:
        return []

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        options: Dict[str, Any] = {}
        if threads is not None:
            options["Threads"] = threads
        if hash_mb is not None:
            options["Hash"] = hash_mb
        if options:
            engine.configure(options)

        infos = engine.analyse(
            board,
            make_limit(depth, time_limit),
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

    scored.sort(key=lambda x: (-10**12 if x.score_cp is None else -x.score_cp, x.move_uci))
    return scored


def eval_gap_to_best(
    stockfish_topk: list[dict[str, Any]],
    scored_candidates: list[CandidateScore],
) -> EvalGap:
    if not stockfish_topk or not scored_candidates:
        return EvalGap(
            best_engine_score_cp=stockfish_topk[0]["score_cp"] if stockfish_topk else None,
            best_candidate_score_cp=None,
            cp_gap=None,
        )

    best_engine = stockfish_topk[0]["score_cp"]
    best_candidate = scored_candidates[0].score_cp
    cp_gap = None if best_engine is None or best_candidate is None else best_engine - best_candidate

    return EvalGap(
        best_engine_score_cp=best_engine,
        best_candidate_score_cp=best_candidate,
        cp_gap=cp_gap,
    )


def safe_mean(values: List[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return mean(clean) if clean else None


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_one(
    json_path: str,
    stockfish_cache_entry: dict[str, Any],
    fast_mode: bool,
    engine_path: str | None,
    depth: int | None,
    time_limit: float | None,
    threads_per_engine: int | None,
    hash_mb: int | None,
) -> Dict[str, Any]:
    path = Path(json_path)
    data = load_json(path)

    fen = data["fen"]
    candidate_moves = data["move_data"]["candidate_moves"]
    topk = stockfish_cache_entry["stockfish_topk"]

    metrics = compare_candidates_to_topk(
        fen=fen,
        candidate_moves=candidate_moves,
        stockfish_topk=topk,
    )

    if fast_mode:
        candidate_scores: list[CandidateScore] = []
        gap = EvalGap(
            best_engine_score_cp=stockfish_cache_entry.get("best_engine_score_cp"),
            best_candidate_score_cp=None,
            cp_gap=None,
        )
    else:
        assert engine_path is not None
        candidate_scores = score_candidate_moves_for_fen(
            engine_path=engine_path,
            fen=fen,
            candidate_moves=candidate_moves,
            depth=depth,
            time_limit=time_limit,
            threads=threads_per_engine,
            hash_mb=hash_mb,
        )
        gap = eval_gap_to_best(topk, candidate_scores)

    return {
        "id": data.get("id", path.stem),
        "file": path.name,
        "fen": fen,
        "candidate_moves": candidate_moves,
        "metrics": asdict(metrics),
        "stockfish_topk": topk,
        "candidate_scores": [asdict(x) for x in candidate_scores],
        "gap_to_best": asdict(gap),
    }


def summarize_folder(folder_name: str, folder_path: Path, per_puzzle: List[dict[str, Any]], elapsed: float, fast_mode: bool) -> dict[str, Any]:
    total_candidate_moves = 0
    total_legal_moves = 0
    total_illegal_moves = 0
    empty_outputs = 0

    recalls: List[float] = []
    precisions: List[float] = []
    jaccards: List[float] = []
    hit_rates: List[int] = []
    top1_hits: List[int] = []
    mrrs: List[float] = []
    cp_gaps: List[int | None] = []
    best_candidate_scores: List[int | None] = []
    best_engine_scores: List[int | None] = []

    for row in per_puzzle:
        m = row["metrics"]
        g = row["gap_to_best"]

        total_candidate_moves += m["num_candidates_unique"]
        total_legal_moves += m["num_candidates_legal"]
        total_illegal_moves += m["num_candidates_illegal"]
        if m["num_candidates_unique"] == 0:
            empty_outputs += 1

        recalls.append(m["recall_at_k"])
        precisions.append(m["precision_at_k"])
        jaccards.append(m["jaccard"])
        hit_rates.append(m["hit_at_k"])
        top1_hits.append(m["oracle_top1_hit"])
        mrrs.append(m["mrr"])
        cp_gaps.append(g["cp_gap"])
        best_candidate_scores.append(g["best_candidate_score_cp"])
        best_engine_scores.append(g["best_engine_score_cp"])

    num_files = len(per_puzzle)
    hallucination_rate = (
        total_illegal_moves / (total_legal_moves + total_illegal_moves)
        if (total_legal_moves + total_illegal_moves) > 0 else None
    )
    empty_output_rate = empty_outputs / num_files if num_files else None

    return {
        "folder_name": folder_name,
        "folder_path": str(folder_path),
        "num_files": num_files,
        "total_candidate_moves_unique": total_candidate_moves,
        "total_legal_moves": total_legal_moves,
        "total_illegal_moves": total_illegal_moves,
        "hallucination_rate": hallucination_rate,
        "empty_output_count": empty_outputs,
        "empty_output_rate": empty_output_rate,
        "avg_recall_at_k": safe_mean(recalls),
        "avg_precision_at_k": safe_mean(precisions),
        "avg_jaccard": safe_mean(jaccards),
        "hit_at_k_rate": safe_mean(hit_rates),
        "top1_hit_rate": safe_mean(top1_hits),
        "avg_mrr": safe_mean(mrrs),
        "avg_cp_gap": safe_mean(cp_gaps),
        "avg_best_candidate_score_cp": safe_mean(best_candidate_scores),
        "avg_best_engine_score_cp": safe_mean(best_engine_scores),
        "elapsed_seconds": elapsed,
        "fast_mode": fast_mode,
    }


def print_compact_table(results: List[dict[str, Any]]) -> None:
    headers = ["folder", "files", "halluc%", "empty%", "R@k", "P@k", "Hit@k", "Top1", "MRR", "avg_cp_gap", "secs"]
    rows = []
    for result in results:
        s = result["summary"]
        rows.append([
            s["folder_name"],
            str(s["num_files"]),
            "NA" if s["hallucination_rate"] is None else f"{100*s['hallucination_rate']:.2f}",
            "NA" if s["empty_output_rate"] is None else f"{100*s['empty_output_rate']:.2f}",
            "NA" if s["avg_recall_at_k"] is None else f"{s['avg_recall_at_k']:.4f}",
            "NA" if s["avg_precision_at_k"] is None else f"{s['avg_precision_at_k']:.4f}",
            "NA" if s["hit_at_k_rate"] is None else f"{s['hit_at_k_rate']:.4f}",
            "NA" if s["top1_hit_rate"] is None else f"{s['top1_hit_rate']:.4f}",
            "NA" if s["avg_mrr"] is None else f"{s['avg_mrr']:.4f}",
            "NA" if s["avg_cp_gap"] is None else f"{s['avg_cp_gap']:.2f}",
            f"{s['elapsed_seconds']:.1f}",
        ])

    widths = [max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate move folders against persisted Stockfish cache.")
    parser.add_argument("--root", default="puzzles/gemma-4-31b-it")
    parser.add_argument("--cache", default="puzzles/stockfish_topk_cache.json")
    parser.add_argument("--folders", nargs="*", default=list(FOLDER_MAP.keys()))
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--engine", default=None, help="Required only when not using --fast")
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-engine", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--tqdm", action="store_true")
    args = parser.parse_args()

    if not args.fast and not args.engine:
        raise ValueError("--engine is required unless using --fast")

    root = Path(args.root)
    cache_path = Path(args.cache)

    with open(cache_path, "r", encoding="utf-8") as f:
        cache_payload = json.load(f)
    cache_positions: Dict[str, Any] = cache_payload["positions"]

    all_results = []
    total_start = time.time()

    for folder_name in args.folders:
        if folder_name not in FOLDER_MAP:
            raise ValueError(f"Unknown folder key: {folder_name}")

        folder_path = root / FOLDER_MAP[folder_name]
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder_path}")

        json_files = sorted(folder_path.glob("*.json"))
        work_items: List[Tuple[str, dict[str, Any]]] = []

        for path in json_files:
            stem = path.stem
            if stem not in cache_positions:
                raise KeyError(f"{stem} missing from cache")
            work_items.append((str(path), cache_positions[stem]))

        folder_start = time.time()
        per_puzzle: List[dict[str, Any]] = []

        print(f"\n=== Evaluating {folder_name} ===")

        iterator = tqdm(total=len(work_items), desc=folder_name, unit="file") if args.tqdm and tqdm is not None else None

        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = {
                ex.submit(
                    process_one,
                    json_path,
                    cache_entry,
                    args.fast,
                    args.engine,
                    args.depth,
                    args.time_limit,
                    args.threads_per_engine,
                    args.hash_mb,
                ): json_path
                for json_path, cache_entry in work_items
            }

            done_count = 0
            for future in as_completed(futures):
                json_path = futures[future]
                stem = Path(json_path).stem
                try:
                    result = future.result()
                    per_puzzle.append(result)
                    done_count += 1
                    if iterator is not None:
                        iterator.update(1)
                    else:
                        if done_count % 10 == 0 or done_count == len(work_items):
                            print(f"[{folder_name}] finished {done_count}/{len(work_items)} in {time.time() - folder_start:.1f}s")
                except Exception as e:
                    if iterator is not None:
                        iterator.update(1)
                    print(json.dumps({"folder": folder_name, "id": stem, "status": "error", "error": str(e)}))

        if iterator is not None:
            iterator.close()

        per_puzzle.sort(key=lambda x: x["id"])
        summary = summarize_folder(
            folder_name=folder_name,
            folder_path=folder_path,
            per_puzzle=per_puzzle,
            elapsed=time.time() - folder_start,
            fast_mode=args.fast,
        )
        all_results.append({
            "summary": summary,
            "per_puzzle": per_puzzle,
        })

    print()
    print_compact_table(all_results)

    payload = {
        "root": str(root),
        "cache": str(cache_path),
        "fast_mode": args.fast,
        "engine": args.engine,
        "depth": args.depth,
        "time_limit": args.time_limit,
        "workers": args.workers,
        "threads_per_engine": args.threads_per_engine,
        "total_elapsed_seconds": time.time() - total_start,
        "results": all_results,
    }

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print()
    print(json.dumps({
        "json_out_written": bool(args.json_out),
        "num_folders_evaluated": len(all_results),
        "total_elapsed_seconds": round(time.time() - total_start, 2),
    }, indent=2))


if __name__ == "__main__":
    main()

# python evaluate_move_folders_from_cache.py --root puzzles/gemma-4-31b-it --cache puzzles/stockfish_topk_cache.json --engine "C:\Users\prvn0\Downloads\stockfish-windows-x86-64-avx2 (1)\stockfish\stockfish-windows-x86-64-avx2.exe" --depth 18 --workers 6 --threads-per-engine 1 --tqdm --json-out puzzles/gemma-4-31b-it/folder_eval_full.json