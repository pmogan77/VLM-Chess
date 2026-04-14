import argparse
import csv
import os
import re
from pathlib import Path
from typing import Optional

import chess
import chess.svg
import cairosvg

from chess_gold_annotations_v1 import (
    generate_gold_annotations,
    render_annotations_to_png,
    save_annotations_json,
)


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text or "unknown"


def is_middlegame(themes: Optional[str]) -> bool:
    if not themes:
        return False
    return "middlegame" in themes.lower().split()


def render_plain_board_png(
    fen: str,
    output_png: str,
    size: int = 720,
    orientation: str = "side_to_move",
) -> None:
    board = chess.Board(fen)

    if orientation == "white":
        board_orientation = chess.WHITE
    elif orientation == "black":
        board_orientation = chess.BLACK
    else:
        board_orientation = board.turn

    svg = chess.svg.board(
        board=board,
        orientation=board_orientation,
        size=size,
        coordinates=True,
    )
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=output_png)


def process_csv(
    csv_path: str,
    output_dir: str,
    size: int = 720,
    orientation: str = "side_to_move",
    limit: Optional[int] = None,
    save_json: bool = True,
) -> None:
    output_root = Path(output_dir)
    plain_dir = output_root / "plain_boards"
    annotated_dir = output_root / "annotated_boards"
    json_dir = output_root / "annotations_json"
    plain_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    if save_json:
        json_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped_non_middlegame = 0
    skipped_invalid = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader, start=1):
            themes = row.get("Themes", "")
            if not is_middlegame(themes):
                skipped_non_middlegame += 1
                continue

            puzzle_id = safe_name(row.get("PuzzleId", f"row_{row_idx}"))
            fen = (row.get("FEN") or "").strip()
            if not fen:
                skipped_invalid += 1
                print(f"[skip] row {row_idx}: missing FEN")
                continue

            try:
                chess.Board(fen)
            except ValueError as exc:
                skipped_invalid += 1
                print(f"[skip] row {row_idx} ({puzzle_id}): invalid FEN -> {exc}")
                continue

            plain_png = plain_dir / f"{puzzle_id}.png"
            annotated_png = annotated_dir / f"{puzzle_id}_annotated.png"
            annotation_json = json_dir / f"{puzzle_id}.json"

            try:
                render_plain_board_png(
                    fen=fen,
                    output_png=str(plain_png),
                    size=size,
                    orientation=orientation,
                )

                annotations = generate_gold_annotations(fen)
                render_annotations_to_png(
                    fen=fen,
                    annotations=annotations,
                    output_png=str(annotated_png),
                    size=size,
                    orientation=orientation,
                )

                if save_json:
                    save_annotations_json(annotations, str(annotation_json))

                processed += 1
                print(f"[ok] {processed}: PuzzleId={puzzle_id}")
            except Exception as exc:
                skipped_invalid += 1
                print(f"[skip] row {row_idx} ({puzzle_id}): render failure -> {exc}")
                continue

            if limit is not None and processed >= limit:
                break

    print("\nDone.")
    print(f"Processed middlegame puzzles: {processed}")
    print(f"Skipped non-middlegame rows: {skipped_non_middlegame}")
    print(f"Skipped invalid rows: {skipped_invalid}")
    print(f"Plain boards: {plain_dir}")
    print(f"Annotated boards: {annotated_dir}")
    if save_json:
        print(f"Annotation JSON: {json_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read a puzzle CSV, keep only middlegame rows, and render plain + annotated chess board PNGs."
    )
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument(
        "--output-dir",
        default="puzzle_renders",
        help="Directory where images and optional JSON files will be written",
    )
    parser.add_argument("--size", type=int, default=720, help="Image size in pixels")
    parser.add_argument(
        "--orientation",
        choices=["side_to_move", "white", "black"],
        default="side_to_move",
        help="Board orientation used for rendering",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of processed middlegame rows",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not save annotation JSON files",
    )
    args = parser.parse_args()

    process_csv(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        size=args.size,
        orientation=args.orientation,
        limit=args.limit,
        save_json=not args.no_json,
    )


if __name__ == "__main__":
    main()
