#!/usr/bin/env python3

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import chess
import chess.svg


# Optional Windows Cairo DLL help.
# Safe to leave in even if you are using WSL/Linux.
if sys.platform == "win32":
    cairo_bin = r"C:\msys64\ucrt64\bin"
    if os.path.isdir(cairo_bin):
        os.environ["CAIROCFFI_DLL_DIRECTORIES"] = cairo_bin
        os.environ["PATH"] = cairo_bin + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(cairo_bin)
        except (AttributeError, FileNotFoundError):
            pass

import cairosvg


# VLM-friendly colors for tan/orange chess boards
COLOR_CANDIDATE_MOVE = "#1e88e5"   # blue
COLOR_THREAT = "#c62828"           # red
COLOR_KEY_SQUARE = "#42a5f566"     # translucent purple


@dataclass
class ArrowAnnotation:
    from_square: str
    to_square: str
    kind: str
    color: str


@dataclass
class SquareAnnotation:
    square: str
    kind: str
    color: str


@dataclass
class AnnotationBundle:
    fen: str
    side_to_move: str
    arrows: List[ArrowAnnotation]
    highlighted_squares: List[SquareAnnotation]


def _square_name(square: chess.Square) -> str:
    return chess.square_name(square)


def _arrow_key(from_sq: chess.Square, to_sq: chess.Square, kind: str) -> Tuple[int, int, str]:
    return from_sq, to_sq, kind


def generate_gold_annotations(
    fen: str,
    max_candidate_moves: Optional[int] = None,
    max_threats_per_target: Optional[int] = None,
    max_total_threats: Optional[int] = None,
    include_quiet_checks: bool = True,
) -> AnnotationBundle:
    """
    Generate deterministic chess annotations.

    Defaults are unlimited:
      max_candidate_moves=None
      max_threats_per_target=None
      max_total_threats=None

    It creates:
      1. Candidate move arrows for side-to-move captures and checks
      2. Opponent threat arrows against side-to-move non-king pieces
      3. Key square highlights for arrow targets

    Optional limits:
      max_candidate_moves = max candidate arrows total
      max_threats_per_target = max threat arrows into one attacked piece
      max_total_threats = max threat arrows across the whole board
    """

    board = chess.Board(fen)
    stm = board.turn
    opp = not stm

    legal_moves = list(board.legal_moves)

    arrows: List[ArrowAnnotation] = []
    arrow_seen: Set[Tuple[int, int, str]] = set()

    square_map: Dict[str, SquareAnnotation] = {}

    def add_arrow(from_sq: chess.Square, to_sq: chess.Square, kind: str, color: str) -> bool:
        key = _arrow_key(from_sq, to_sq, kind)

        if key in arrow_seen:
            return False

        arrow_seen.add(key)

        arrows.append(
            ArrowAnnotation(
                from_square=_square_name(from_sq),
                to_square=_square_name(to_sq),
                kind=kind,
                color=color,
            )
        )

        return True

    def add_key_square(square: chess.Square) -> None:
        sq_name = _square_name(square)

        if sq_name not in square_map:
            square_map[sq_name] = SquareAnnotation(
                square=sq_name,
                kind="key_square",
                color=COLOR_KEY_SQUARE,
            )

    # 1. Candidate move arrows
    # None means no limit.
    tactical_moves: List[chess.Move] = []

    for move in legal_moves:
        is_capture = board.is_capture(move)
        gives_check = board.gives_check(move)

        if is_capture or (include_quiet_checks and gives_check):
            tactical_moves.append(move)

    tactical_moves.sort(
        key=lambda move: (
            0 if board.is_capture(move) else 1,
            chess.square_name(move.from_square),
            chess.square_name(move.to_square),
        )
    )

    if max_candidate_moves is not None:
        tactical_moves = tactical_moves[:max_candidate_moves]

    for move in tactical_moves:
        added = add_arrow(
            from_sq=move.from_square,
            to_sq=move.to_square,
            kind="candidate_move",
            color=COLOR_CANDIDATE_MOVE,
        )

        if added:
            add_key_square(move.to_square)

    # 2. Opponent threat arrows
    # None means no limit.
    total_threats_added = 0

    for target_sq in chess.SQUARES:
        if max_total_threats is not None and total_threats_added >= max_total_threats:
            break

        piece = board.piece_at(target_sq)

        if piece is None:
            continue

        # Only show threats against side-to-move pieces.
        if piece.color != stm:
            continue

        # # Skip king threats to avoid clutter and weird check semantics.
        # if piece.piece_type == chess.KING:
        #     continue

        attackers = sorted(board.attackers(opp, target_sq))

        if max_threats_per_target is not None:
            attackers = attackers[:max_threats_per_target]

        for attacker_sq in attackers:
            if max_total_threats is not None and total_threats_added >= max_total_threats:
                break

            added = add_arrow(
                from_sq=attacker_sq,
                to_sq=target_sq,
                kind="threat",
                color=COLOR_THREAT,
            )

            if added:
                add_key_square(target_sq)
                total_threats_added += 1

    arrows.sort(
        key=lambda arrow: (
            chess.parse_square(arrow.from_square),
            chess.parse_square(arrow.to_square),
            arrow.kind,
        )
    )

    highlighted_squares = sorted(
        square_map.values(),
        key=lambda square_annotation: chess.parse_square(square_annotation.square),
    )

    return AnnotationBundle(
        fen=fen,
        side_to_move="white" if stm == chess.WHITE else "black",
        arrows=arrows,
        highlighted_squares=highlighted_squares,
    )


def render_annotations_to_png(
    fen: str,
    annotations: AnnotationBundle,
    output_png: str,
    size: int = 720,
    orientation: Optional[str] = "side_to_move",
    arrow_stroke_px: int = 5,
    arrowhead_scale: float = 0.5,
) -> None:
    """
    Render an AnnotationBundle to a PNG chess board.
    """

    import xml.etree.ElementTree as ET

    board = chess.Board(fen)

    svg_arrows = [
        chess.svg.Arrow(
            tail=chess.parse_square(annotation.from_square),
            head=chess.parse_square(annotation.to_square),
            color=annotation.color,
        )
        for annotation in annotations.arrows
    ]

    fill = {
        chess.parse_square(square_annotation.square): square_annotation.color
        for square_annotation in annotations.highlighted_squares
    }

    if orientation == "white":
        board_orientation = chess.WHITE
    elif orientation == "black":
        board_orientation = chess.BLACK
    else:
        board_orientation = board.turn

    svg = chess.svg.board(
        board=board,
        orientation=board_orientation,
        arrows=svg_arrows,
        fill=fill,
        size=size,
        coordinates=True,
        style=f".arrow {{ stroke-width: {arrow_stroke_px}px; }}",
    )

    root = ET.fromstring(svg)

    def has_arrow_class(elem: ET.Element) -> bool:
        classes = elem.attrib.get("class", "").split()
        return "arrow" in classes

    def scale_arrowhead_points(points_str: str, scale: float) -> str:
        points = []

        for token in points_str.strip().split():
            x_str, y_str = token.split(",")
            points.append((float(x_str), float(y_str)))

        if len(points) != 3:
            return points_str

        tip = points[0]
        base1 = points[1]
        base2 = points[2]

        anchor_x = (base1[0] + base2[0]) / 2.0
        anchor_y = (base1[1] + base2[1]) / 2.0

        scaled = []

        for x, y in (tip, base1, base2):
            sx = anchor_x + scale * (x - anchor_x)
            sy = anchor_y + scale * (y - anchor_y)
            scaled.append(f"{sx:.3f},{sy:.3f}")

        return " ".join(scaled)

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]

        if tag == "polygon" and has_arrow_class(elem):
            points = elem.attrib.get("points")
            if points:
                elem.set("points", scale_arrowhead_points(points, arrowhead_scale))

    svg_out = ET.tostring(root, encoding="unicode")

    cairosvg.svg2png(
        bytestring=svg_out.encode("utf-8"),
        write_to=output_png,
    )


def save_annotations_json(annotations: AnnotationBundle, output_json: str) -> None:
    """
    Save annotations to JSON.
    """

    data = {
        "fen": annotations.fen,
        "side_to_move": annotations.side_to_move,
        "arrows": [asdict(arrow) for arrow in annotations.arrows],
        "highlighted_squares": [
            asdict(square_annotation)
            for square_annotation in annotations.highlighted_squares
        ],
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def process_one_fen(
    fen: str,
    output_json: str,
    output_png: str,
    size: int = 720,
    orientation: str = "side_to_move",
    max_candidate_moves: Optional[int] = None,
    max_threats_per_target: Optional[int] = None,
    max_total_threats: Optional[int] = None,
    include_quiet_checks: bool = True,
) -> AnnotationBundle:
    """
    Generate, save, and render annotations for one FEN.
    """

    annotations = generate_gold_annotations(
        fen=fen,
        max_candidate_moves=max_candidate_moves,
        max_threats_per_target=max_threats_per_target,
        max_total_threats=max_total_threats,
        include_quiet_checks=include_quiet_checks,
    )

    save_annotations_json(annotations, output_json)

    render_annotations_to_png(
        fen=fen,
        annotations=annotations,
        output_png=output_png,
        size=size,
        orientation=orientation,
    )

    return annotations


def process_in_place(
    json_dir: Path,
    png_dir: Path,
    size: int = 720,
    orientation: str = "side_to_move",
    max_candidate_moves: Optional[int] = None,
    max_threats_per_target: Optional[int] = None,
    max_total_threats: Optional[int] = None,
    include_quiet_checks: bool = True,
    overwrite_missing_png_only: bool = False,
) -> None:
    """
    Rewrite existing annotation JSON files in place.
    Also overwrite the matching PNG file.

    By default, no arrow limits are imposed.
    """

    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return

    processed = 0
    skipped = 0
    errors = 0

    png_dir.mkdir(parents=True, exist_ok=True)

    for json_path in json_files:
        stem = json_path.stem
        png_path = png_dir / f"{stem}.png"

        try:
            if overwrite_missing_png_only and png_path.exists():
                skipped += 1
                print(json.dumps({"id": stem, "status": "skipped_png_exists"}))
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            fen = data.get("fen")

            if not fen:
                errors += 1
                print(json.dumps({"id": stem, "status": "error", "error": "missing_fen"}))
                continue

            annotations = generate_gold_annotations(
                fen=fen,
                max_candidate_moves=max_candidate_moves,
                max_threats_per_target=max_threats_per_target,
                max_total_threats=max_total_threats,
                include_quiet_checks=include_quiet_checks,
            )

            save_annotations_json(annotations, str(json_path))

            render_annotations_to_png(
                fen=fen,
                annotations=annotations,
                output_png=str(png_path),
                size=size,
                orientation=orientation,
            )

            processed += 1

            print(json.dumps({
                "id": stem,
                "status": "ok",
                "json": str(json_path),
                "png": str(png_path),
                "arrows": len(annotations.arrows),
                "highlighted_squares": len(annotations.highlighted_squares),
            }))

        except Exception as e:
            errors += 1
            print(json.dumps({
                "id": stem,
                "status": "error",
                "error": str(e),
            }))

    print(json.dumps({
        "json_dir": str(json_dir),
        "png_dir": str(png_dir),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
    }, indent=2))


def _none_or_int(value: Optional[str]) -> Optional[int]:
    """
    argparse helper.

    Allows:
      --max-candidate-moves 3
      --max-candidate-moves none
      omitted argument means None by default
    """

    if value is None:
        return None

    cleaned = value.strip().lower()

    if cleaned in {"none", "null", "unlimited", "all", "-1"}:
        return None

    parsed = int(cleaned)

    if parsed < 0:
        return None

    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic chess annotations and render PNG boards."
    )

    parser.add_argument(
        "fen",
        nargs="?",
        help="Optional single FEN string. If omitted, use --json-dir batch mode.",
    )

    parser.add_argument(
        "--json",
        default="annotations.json",
        help="Output JSON path for single-FEN mode.",
    )

    parser.add_argument(
        "--png",
        default="annotated_board.png",
        help="Output PNG path for single-FEN mode.",
    )

    parser.add_argument(
        "--json-dir",
        default=None,
        help="Directory of existing annotation JSON files to rewrite in place.",
    )

    parser.add_argument(
        "--png-dir",
        default=None,
        help="Directory where matching PNG files should be written.",
    )

    parser.add_argument("--size", type=int, default=720)

    parser.add_argument(
        "--orientation",
        choices=["side_to_move", "white", "black"],
        default="side_to_move",
    )

    parser.add_argument(
        "--max-candidate-moves",
        type=_none_or_int,
        default=None,
        help="Max candidate arrows. Default None means unlimited.",
    )

    parser.add_argument(
        "--max-threats-per-target",
        type=_none_or_int,
        default=None,
        help="Max threat arrows into one attacked piece. Default None means unlimited.",
    )

    parser.add_argument(
        "--max-total-threats",
        type=_none_or_int,
        default=None,
        help="Max threat arrows across the whole board. Default None means unlimited.",
    )

    parser.add_argument(
        "--no-quiet-checks",
        action="store_true",
        help="Exclude non-capture checking moves from candidate arrows.",
    )

    parser.add_argument(
        "--overwrite-missing-png-only",
        action="store_true",
        help="Only process entries whose PNG is missing.",
    )

    args = parser.parse_args()

    include_quiet_checks = not args.no_quiet_checks

    # Batch rewrite mode
    if args.json_dir is not None:
        json_dir = Path(args.json_dir)

        if args.png_dir is None:
            raise ValueError("--png-dir is required when using --json-dir")

        png_dir = Path(args.png_dir)

        if not json_dir.is_dir():
            raise FileNotFoundError(f"Missing JSON directory: {json_dir}")

        process_in_place(
            json_dir=json_dir,
            png_dir=png_dir,
            size=args.size,
            orientation=args.orientation,
            max_candidate_moves=args.max_candidate_moves,
            max_threats_per_target=args.max_threats_per_target,
            max_total_threats=args.max_total_threats,
            include_quiet_checks=include_quiet_checks,
            overwrite_missing_png_only=args.overwrite_missing_png_only,
        )

        return

    # Single FEN mode
    if not args.fen:
        raise ValueError("Provide a FEN string or use --json-dir batch mode.")

    annotations = process_one_fen(
        fen=args.fen,
        output_json=args.json,
        output_png=args.png,
        size=args.size,
        orientation=args.orientation,
        max_candidate_moves=args.max_candidate_moves,
        max_threats_per_target=args.max_threats_per_target,
        max_total_threats=args.max_total_threats,
        include_quiet_checks=include_quiet_checks,
    )

    print(f"Saved JSON to {args.json}")
    print(f"Saved PNG to {args.png}")
    print(f"Arrows: {len(annotations.arrows)}")
    print(f"Highlighted squares: {len(annotations.highlighted_squares)}")


if __name__ == "__main__":
    main()

# python gold_arrow_generation/chess_gold_annotations.py --json-dir puzzles/annotations_json --png-dir puzzles/annotated_boards --size 720 --orientation side_to_move