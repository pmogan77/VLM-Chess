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


# Simplified visual vocabulary
COLOR_CANDIDATE_MOVE = "#2e7d32"   # green
COLOR_THREAT = "#fb8c00"           # orange
COLOR_KEY_SQUARE = "#42a5f566"     # blue translucent


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
    return (from_sq, to_sq, kind)


def generate_minimal_annotations(
    fen: str,
    max_candidate_moves: int = 3,
    max_threats_per_target: int = 2,
    include_quiet_checks: bool = True,
) -> AnnotationBundle:
    board = chess.Board(fen)
    stm = board.turn
    opp = not stm

    legal_moves = list(board.legal_moves)

    arrows: List[ArrowAnnotation] = []
    arrow_seen: Set[Tuple[int, int, str]] = set()

    square_map: Dict[str, SquareAnnotation] = {}

    def add_arrow(from_sq: chess.Square, to_sq: chess.Square, kind: str, color: str) -> None:
        key = _arrow_key(from_sq, to_sq, kind)
        if key in arrow_seen:
            return
        arrow_seen.add(key)
        arrows.append(
            ArrowAnnotation(
                from_square=_square_name(from_sq),
                to_square=_square_name(to_sq),
                kind=kind,
                color=color,
            )
        )

    def add_key_square(square: chess.Square) -> None:
        sq_name = _square_name(square)
        if sq_name not in square_map:
            square_map[sq_name] = SquareAnnotation(
                square=sq_name,
                kind="key_square",
                color=COLOR_KEY_SQUARE,
            )

    # 1) candidate moves
    tactical_moves: List[chess.Move] = []
    for move in legal_moves:
        is_capture = board.is_capture(move)
        gives_check = board.gives_check(move)
        if is_capture or (include_quiet_checks and gives_check):
            tactical_moves.append(move)

    tactical_moves.sort(
        key=lambda m: (
            0 if board.is_capture(m) else 1,
            chess.square_name(m.from_square),
            chess.square_name(m.to_square),
        )
    )

    for move in tactical_moves[:max_candidate_moves]:
        add_arrow(move.from_square, move.to_square, "candidate_move", COLOR_CANDIDATE_MOVE)
        add_key_square(move.to_square)

    # 2) threats from opponent onto side-to-move pieces
    for target_sq in chess.SQUARES:
        piece = board.piece_at(target_sq)
        if piece is None or piece.color != stm:
            continue
        if piece.piece_type == chess.KING:
            continue

        attackers = sorted(board.attackers(opp, target_sq))
        for attacker_sq in attackers[:max_threats_per_target]:
            add_arrow(attacker_sq, target_sq, "threat", COLOR_THREAT)
            add_key_square(target_sq)

    arrows.sort(
        key=lambda a: (
            chess.parse_square(a.from_square),
            chess.parse_square(a.to_square),
            a.kind,
        )
    )
    highlighted_squares = sorted(square_map.values(), key=lambda x: chess.parse_square(x.square))

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
    arrow_stroke_px: int = 8,
    arrowhead_scale: float = 0.72,
) -> None:
    import xml.etree.ElementTree as ET

    board = chess.Board(fen)

    svg_arrows = [
        chess.svg.Arrow(
            chess.parse_square(a.from_square),
            chess.parse_square(a.to_square),
            color=a.color,
        )
        for a in annotations.arrows
    ]

    fill = {
        chess.parse_square(s.square): s.color
        for s in annotations.highlighted_squares
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
        pts = []
        for token in points_str.strip().split():
            x_str, y_str = token.split(",")
            pts.append((float(x_str), float(y_str)))

        if len(pts) != 3:
            return points_str

        tip = pts[0]
        base1 = pts[1]
        base2 = pts[2]

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
    cairosvg.svg2png(bytestring=svg_out.encode("utf-8"), write_to=output_png)


def save_annotations_json(annotations: AnnotationBundle, output_json: str) -> None:
    data = {
        "fen": annotations.fen,
        "side_to_move": annotations.side_to_move,
        "arrows": [asdict(a) for a in annotations.arrows],
        "highlighted_squares": [asdict(s) for s in annotations.highlighted_squares],
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def process_in_place(
    json_dir: Path,
    png_dir: Path,
    size: int,
    orientation: str,
    max_candidate_moves: int,
    max_threats_per_target: int,
    include_quiet_checks: bool,
    overwrite_missing_png_only: bool = False,
) -> None:
    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return

    processed = 0
    skipped = 0
    errors = 0

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

            annotations = generate_minimal_annotations(
                fen=fen,
                max_candidate_moves=max_candidate_moves,
                max_threats_per_target=max_threats_per_target,
                include_quiet_checks=include_quiet_checks,
            )

            # overwrite JSON in place
            save_annotations_json(annotations, str(json_path))

            # overwrite PNG in place
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
            print(json.dumps({"id": stem, "status": "error", "error": str(e)}))

    print(json.dumps({
        "json_dir": str(json_dir),
        "png_dir": str(png_dir),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite chess annotation JSON files in place and overwrite matching PNG boards."
    )
    parser.add_argument(
        "--json-dir",
        default="puzzles/annotations_json",
        help="Directory containing per-board annotation JSON files",
    )
    parser.add_argument(
        "--png-dir",
        default="puzzles/annotated_boards",
        help="Directory containing matching PNG files to overwrite",
    )
    parser.add_argument("--size", type=int, default=720)
    parser.add_argument(
        "--orientation",
        choices=["side_to_move", "white", "black"],
        default="side_to_move",
    )
    parser.add_argument("--max-candidate-moves", type=int, default=3)
    parser.add_argument("--max-threats-per-target", type=int, default=2)
    parser.add_argument("--no-quiet-checks", action="store_true")
    parser.add_argument(
        "--overwrite-missing-png-only",
        action="store_true",
        help="Only process entries whose PNG is missing",
    )
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    png_dir = Path(args.png_dir)

    if not json_dir.is_dir():
        raise FileNotFoundError(f"Missing JSON directory: {json_dir}")
    if not png_dir.is_dir():
        raise FileNotFoundError(f"Missing PNG directory: {png_dir}")

    process_in_place(
        json_dir=json_dir,
        png_dir=png_dir,
        size=args.size,
        orientation=args.orientation,
        max_candidate_moves=args.max_candidate_moves,
        max_threats_per_target=args.max_threats_per_target,
        include_quiet_checks=not args.no_quiet_checks,
        overwrite_missing_png_only=args.overwrite_missing_png_only,
    )


if __name__ == "__main__":
    main()