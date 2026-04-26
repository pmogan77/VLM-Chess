#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import chess
import chess.engine

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


MATE_SCORE = 100000


@dataclass
class EngineMove:
    rank: int
    move_uci: str
    pv: list[str]
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


def get_stockfish_top_k_for_fen(
    engine_path: str,
    fen: str,
    k: int,
    depth: int | None,
    time_limit: float | None,
    threads: int | None,
    hash_mb: int | None,
) -> list[EngineMove]:
    board = chess.Board(fen)
    limit = make_limit(depth, time_limit)

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


def process_one(
    json_path: str,
    engine_path: str,
    k: int,
    depth: int | None,
    time_limit: float | None,
    threads: int | None,
    hash_mb: int | None,
) -> Dict[str, Any]:
    path = Path(json_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fen = data.get("fen")
    if not fen:
        raise ValueError(f"Missing fen in {path}")

    topk = get_stockfish_top_k_for_fen(
        engine_path=engine_path,
        fen=fen,
        k=k,
        depth=depth,
        time_limit=time_limit,
        threads=threads,
        hash_mb=hash_mb,
    )

    return {
        "id": path.stem,
        "fen": fen,
        "stockfish_topk": [asdict(x) for x in topk],
        "best_engine_score_cp": topk[0].score_cp if topk else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build persistent Stockfish top-k cache from annotations_json.")
    parser.add_argument("--annotations-dir", default="puzzles/annotations_json")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--cache-out", default="puzzles/stockfish_topk_cache.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-engine", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tqdm", action="store_true")
    args = parser.parse_args()

    annotations_dir = Path(args.annotations_dir)
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Missing annotations dir: {annotations_dir}")

    cache_out = Path(args.cache_out)

    existing_cache: Dict[str, Any] = {}
    if cache_out.exists() and not args.force:
        with open(cache_out, "r", encoding="utf-8") as f:
            existing_cache = json.load(f)

    existing_positions = existing_cache.get("positions", {})
    json_files = sorted(annotations_dir.glob("*.json"))

    work = [str(p) for p in json_files if p.stem not in existing_positions]

    start = time.time()
    if args.tqdm and tqdm is not None:
        iterator = tqdm(total=len(work), desc="build_cache", unit="pos")
    else:
        iterator = None

    results: Dict[str, Any] = dict(existing_positions)

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {
            ex.submit(
                process_one,
                json_path,
                args.engine,
                args.k,
                args.depth,
                args.time_limit,
                args.threads_per_engine,
                args.hash_mb,
            ): json_path
            for json_path in work
        }

        done_count = 0
        for future in as_completed(futures):
            json_path = futures[future]
            stem = Path(json_path).stem
            try:
                result = future.result()
                results[stem] = result
                done_count += 1
                if iterator is not None:
                    iterator.update(1)
                else:
                    print(f"[cache] finished {done_count}/{len(work)}: {stem}")
            except Exception as e:
                if iterator is not None:
                    iterator.update(1)
                print(json.dumps({"id": stem, "status": "error", "error": str(e)}))

    if iterator is not None:
        iterator.close()

    payload = {
        "engine": args.engine,
        "k": args.k,
        "depth": args.depth,
        "time_limit": args.time_limit,
        "positions": dict(sorted(results.items())),
        "num_positions": len(results),
        "elapsed_seconds": time.time() - start,
    }

    cache_out.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps({
        "cache_out": str(cache_out),
        "num_positions": len(results),
        "elapsed_seconds": round(time.time() - start, 2),
    }, indent=2))


if __name__ == "__main__":
    main()

# python build_stockfish_cache.py --annotations-dir puzzles/annotations_json --engine "C:\Users\prvn0\Downloads\stockfish-windows-x86-64-avx2 (1)\stockfish\stockfish-windows-x86-64-avx2.exe" --cache-out puzzles/stockfish_topk_cache.json --k 5 --depth 18 --workers 6 --threads-per-engine 1 --tqdm