#!/usr/bin/env python3
"""
Random baseline for chess annotations.

Reads:
    puzzles/annotations_json/<id>.json

Each input JSON must contain:
    "fen"

Writes:
    puzzles/random/annotations_json/<id>.json
    puzzles/random/annotated_boards/<id>.png

This baseline does NOT read board PNGs.
It uses only the FEN and python-chess to generate random annotations.

Rendering is shared with:
    gold_arrow_generation.chess_gold_annotations.render_annotations_to_png
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List

import chess

from gold_arrow_generation.chess_gold_annotations import (
    AnnotationBundle,
    ArrowAnnotation,
    SquareAnnotation,
    COLOR_CANDIDATE_MOVE,
    COLOR_THREAT,
    COLOR_KEY_SQUARE,
    render_annotations_to_png,
)


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []

    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)

    return out


def arrow_record_key(item: Dict[str, str]) -> str:
    return f'{item.get("from", "")}::{item.get("to", "")}::{item.get("piece", "")}'


def dedupe_arrow_records(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []

    for item in items:
        key = arrow_record_key(item)

        if key not in seen:
            seen.add(key)
            out.append(item)

    return out


def validate_key_squares(items: List[str]) -> List[str]:
    valid = []

    for sq in dedupe_keep_order(items):
        try:
            chess.parse_square(sq)
            valid.append(sq)
        except ValueError:
            pass

    return valid


def sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)

    return name or "random"


def build_actual_threat_arrows(board: chess.Board) -> List[Dict[str, str]]:
    """
    Threat arrows are sampled from real opponent attack relations.

    Priority:
      1. Opponent attacks on side-to-move pieces, including king
      2. Opponent attacks on squares near side-to-move king
      3. Opponent attacks on central squares
    """

    my_color = board.turn
    opponent_color = not board.turn

    my_piece_map = {
        sq: piece
        for sq, piece in board.piece_map().items()
        if piece.color == my_color
    }

    opp_piece_map = {
        sq: piece
        for sq, piece in board.piece_map().items()
        if piece.color == opponent_color
    }

    arrows: List[Dict[str, str]] = []
    seen = set()

    my_king_sq = board.king(my_color)

    king_ring = set()

    if my_king_sq is not None:
        king_file = chess.square_file(my_king_sq)
        king_rank = chess.square_rank(my_king_sq)

        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                file_idx = king_file + df
                rank_idx = king_rank + dr

                if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
                    king_ring.add(chess.square(file_idx, rank_idx))

    center_squares = {
        chess.D4, chess.E4, chess.D5, chess.E5,
        chess.C3, chess.F3, chess.C6, chess.F6,
        chess.C4, chess.F4, chess.C5, chess.F5,
    }

    # 1. Opponent attacks on my pieces, including king.
    for target_sq in my_piece_map:
        attackers = board.attackers(opponent_color, target_sq)

        for attacker_sq in attackers:
            piece = board.piece_at(attacker_sq)

            if piece is None:
                continue

            item = {
                "from": chess.square_name(attacker_sq),
                "to": chess.square_name(target_sq),
                "piece": piece.symbol(),
            }

            key = arrow_record_key(item)

            if key not in seen:
                seen.add(key)
                arrows.append(item)

    # 2. Opponent attacks near my king.
    for from_sq, piece in opp_piece_map.items():
        for to_sq in board.attacks(from_sq):
            if to_sq in king_ring:
                item = {
                    "from": chess.square_name(from_sq),
                    "to": chess.square_name(to_sq),
                    "piece": piece.symbol(),
                }

                key = arrow_record_key(item)

                if key not in seen:
                    seen.add(key)
                    arrows.append(item)

    # 3. Opponent attacks central squares.
    for from_sq, piece in opp_piece_map.items():
        for to_sq in board.attacks(from_sq):
            if to_sq in center_squares:
                item = {
                    "from": chess.square_name(from_sq),
                    "to": chess.square_name(to_sq),
                    "piece": piece.symbol(),
                }

                key = arrow_record_key(item)

                if key not in seen:
                    seen.add(key)
                    arrows.append(item)

    return arrows


def sample_random_legal_moves(
    board: chess.Board,
    rng: random.Random,
    max_candidate_arrows: int,
) -> List[Dict[str, str]]:
    legal_moves = list(board.legal_moves)
    rng.shuffle(legal_moves)

    out: List[Dict[str, str]] = []

    for move in legal_moves[:max_candidate_arrows]:
        piece = board.piece_at(move.from_square)

        if piece is None:
            continue

        out.append(
            {
                "from": chess.square_name(move.from_square),
                "to": chess.square_name(move.to_square),
                "piece": piece.symbol(),
            }
        )

    return dedupe_arrow_records(out)


def sample_random_threats(
    board: chess.Board,
    rng: random.Random,
    max_threat_arrows: int,
) -> List[Dict[str, str]]:
    all_threats = build_actual_threat_arrows(board)
    rng.shuffle(all_threats)

    return dedupe_arrow_records(all_threats[:max_threat_arrows])


def sample_random_key_squares(
    rng: random.Random,
    max_key_squares: int,
) -> List[str]:
    all_squares = [chess.square_name(sq) for sq in chess.SQUARES]
    rng.shuffle(all_squares)

    return validate_key_squares(all_squares[:max_key_squares])


def build_random_annotation_bundle(
    fen: str,
    side_to_move: str,
    candidate_arrows: List[Dict[str, str]],
    threat_arrows: List[Dict[str, str]],
    key_squares: List[str],
) -> AnnotationBundle:
    """
    Convert random baseline dict output into the same AnnotationBundle
    used by the gold annotation renderer.
    """

    arrows: List[ArrowAnnotation] = []

    for item in dedupe_arrow_records(candidate_arrows):
        arrows.append(
            ArrowAnnotation(
                from_square=item["from"],
                to_square=item["to"],
                kind="candidate_move",
                color=COLOR_CANDIDATE_MOVE,
            )
        )

    for item in dedupe_arrow_records(threat_arrows):
        arrows.append(
            ArrowAnnotation(
                from_square=item["from"],
                to_square=item["to"],
                kind="threat",
                color=COLOR_THREAT,
            )
        )

    highlighted_squares = [
        SquareAnnotation(
            square=sq,
            kind="key_square",
            color=COLOR_KEY_SQUARE,
        )
        for sq in validate_key_squares(key_squares)
    ]

    return AnnotationBundle(
        fen=fen,
        side_to_move=side_to_move,
        arrows=arrows,
        highlighted_squares=highlighted_squares,
    )


def process_one(
    input_annotation_json_path: Path,
    output_annotation_json_path: Path,
    output_final_png_path: Path,
    model_name: str,
    max_candidate_arrows: int,
    max_threat_arrows: int,
    max_key_squares: int,
    size: int,
    orientation: str,
    global_seed: int,
) -> Dict:
    with open(input_annotation_json_path, "r", encoding="utf-8") as f:
        input_meta = json.load(f)

    fen = input_meta.get("fen")

    if not fen:
        raise ValueError(f"Missing top-level 'fen' in {input_annotation_json_path}")

    board = chess.Board(fen)
    side_to_move = "white" if board.turn == chess.WHITE else "black"

    rng = random.Random(f"{global_seed}:{input_annotation_json_path.stem}")

    candidate_move_arrows = sample_random_legal_moves(
        board=board,
        rng=rng,
        max_candidate_arrows=max_candidate_arrows,
    )

    threat_arrows = sample_random_threats(
        board=board,
        rng=rng,
        max_threat_arrows=max_threat_arrows,
    )

    key_squares = sample_random_key_squares(
        rng=rng,
        max_key_squares=max_key_squares,
    )

    annotations = build_random_annotation_bundle(
        fen=fen,
        side_to_move=side_to_move,
        candidate_arrows=candidate_move_arrows,
        threat_arrows=threat_arrows,
        key_squares=key_squares,
    )

    render_annotations_to_png(
        fen=fen,
        annotations=annotations,
        output_png=str(output_final_png_path),
        size=size,
        orientation=orientation,
    )

    parsed_output = {
        "id": input_annotation_json_path.stem,
        "input_annotation_json_path": str(input_annotation_json_path),
        "side_to_move": side_to_move,
        "fen": fen,
        "model": model_name,
        "baseline_type": "random_baseline",
        "baseline_seed": f"{global_seed}:{input_annotation_json_path.stem}",
        "candidate_move_arrows": candidate_move_arrows,
        "threat_arrows": threat_arrows,
        "key_squares": key_squares,
        "annotation_bundle": {
            "arrows": [
                {
                    "from_square": arrow.from_square,
                    "to_square": arrow.to_square,
                    "kind": arrow.kind,
                    "color": arrow.color,
                }
                for arrow in annotations.arrows
            ],
            "highlighted_squares": [
                {
                    "square": square.square,
                    "kind": square.kind,
                    "color": square.color,
                }
                for square in annotations.highlighted_squares
            ],
        },
        "final_png": str(output_final_png_path),
    }

    with open(output_annotation_json_path, "w", encoding="utf-8") as f:
        json.dump(parsed_output, f, indent=2)

    return parsed_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random chess annotation baseline from FEN only."
    )

    parser.add_argument(
        "--puzzles-dir",
        required=True,
        help="Path to root puzzles directory.",
    )

    parser.add_argument(
        "--output-name",
        default="random",
        help="Output folder name inside puzzles.",
    )

    parser.add_argument("--max-candidate-arrows", type=int, default=3)
    parser.add_argument("--max-threat-arrows", type=int, default=3)
    parser.add_argument("--max-key-squares", type=int, default=3)
    parser.add_argument("--size", type=int, default=720)

    parser.add_argument(
        "--orientation",
        choices=["side_to_move", "white", "black"],
        default="side_to_move",
    )

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    puzzles_dir = Path(args.puzzles_dir)
    input_annotations_dir = puzzles_dir / "annotations_json"

    if not input_annotations_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {input_annotations_dir}")

    model_name = sanitize_name(args.output_name)
    model_dir = puzzles_dir / model_name

    output_annotations_dir = model_dir / "annotations_json"
    output_final_dir = model_dir / "annotated_boards"

    output_annotations_dir.mkdir(parents=True, exist_ok=True)
    output_final_dir.mkdir(parents=True, exist_ok=True)

    input_json_paths = sorted(input_annotations_dir.glob("*.json"))

    if args.start_index:
        input_json_paths = input_json_paths[args.start_index:]

    if args.limit > 0:
        input_json_paths = input_json_paths[:args.limit]

    processed = 0
    errors = []

    for input_annotation_json_path in input_json_paths:
        stem = input_annotation_json_path.stem

        output_annotation_json_path = output_annotations_dir / f"{stem}.json"
        output_final_png_path = output_final_dir / f"{stem}.png"

        if (
            not args.force
            and output_annotation_json_path.exists()
            and output_final_png_path.exists()
        ):
            print(json.dumps({
                "id": stem,
                "status": "skipped_complete",
                "final_png": str(output_final_png_path),
                "json": str(output_annotation_json_path),
            }))
            continue

        try:
            result = process_one(
                input_annotation_json_path=input_annotation_json_path,
                output_annotation_json_path=output_annotation_json_path,
                output_final_png_path=output_final_png_path,
                model_name=model_name,
                max_candidate_arrows=args.max_candidate_arrows,
                max_threat_arrows=args.max_threat_arrows,
                max_key_squares=args.max_key_squares,
                size=args.size,
                orientation=args.orientation,
                global_seed=args.seed,
            )

            processed += 1

            print(json.dumps({
                "id": stem,
                "status": "ok",
                "final_png": str(output_final_png_path),
                "json": str(output_annotation_json_path),
                "candidate_move_arrows": len(result["candidate_move_arrows"]),
                "threat_arrows": len(result["threat_arrows"]),
                "key_squares": len(result["key_squares"]),
            }))

        except Exception as e:
            errors.append({"id": stem, "error": str(e)})
            print(json.dumps({
                "id": stem,
                "status": "error",
                "error": str(e),
            }))

    summary = {
        "puzzles_dir": str(puzzles_dir),
        "model_output_dir": str(model_dir),
        "processed": processed,
        "errors": len(errors),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

# python random_baseline.py --puzzles-dir puzzles --output-name random --max-candidate-arrows 3 --max-threat-arrows 3 --max-key-squares 3 --size 720 --orientation side_to_move --force