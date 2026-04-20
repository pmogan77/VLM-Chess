import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
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


COLOR_CAPTURE = "#d32f2f"              # red
COLOR_CHECK = "#1976d2"                # blue
COLOR_CAPTURE_CHECK = "#7b1fa2"        # purple
COLOR_HANGING = "#ffb74d88"            # orange translucent
COLOR_CAPTURE_TARGET = "#ef9a9a66"
COLOR_CHECKED_KING = "#e53935aa"
COLOR_CHECKER = "#9575cd88"

COLOR_OPP_PRESSURE = "#fb8c00"         # orange
COLOR_OPP_PRESSURE_TARGET = "#ffcc8066"
COLOR_PINNED = "#8e24aa66"             # violet translucent
COLOR_PIN_ARROW = "#8e24aa"            # violet


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


def _is_hanging_piece(board: chess.Board, square: chess.Square, attacker_color: chess.Color) -> bool:
    piece = board.piece_at(square)
    if piece is None:
        return False
    if piece.color == attacker_color:
        return False
    if piece.piece_type == chess.KING:
        return False

    attackers = board.attackers(attacker_color, square)
    defenders = board.attackers(piece.color, square)
    return bool(attackers) and len(defenders) == 0


def _aligned(a: chess.Square, b: chess.Square) -> bool:
    af, ar = chess.square_file(a), chess.square_rank(a)
    bf, br = chess.square_file(b), chess.square_rank(b)
    return af == bf or ar == br or abs(af - bf) == abs(ar - br)


def _step_from_to(a: chess.Square, b: chess.Square) -> Optional[Tuple[int, int]]:
    if not _aligned(a, b):
        return None

    af, ar = chess.square_file(a), chess.square_rank(a)
    bf, br = chess.square_file(b), chess.square_rank(b)

    df = 0 if bf == af else (1 if bf > af else -1)
    dr = 0 if br == ar else (1 if br > ar else -1)
    return df, dr


def _find_pinner(board: chess.Board, color: chess.Color, square: chess.Square) -> Optional[chess.Square]:
    king_square = board.king(color)
    if king_square is None:
        return None

    if not board.is_pinned(color, square):
        return None

    step = _step_from_to(king_square, square)
    if step is None:
        return None

    df, dr = step
    f = chess.square_file(square) + df
    r = chess.square_rank(square) + dr

    while 0 <= f < 8 and 0 <= r < 8:
        sq = chess.square(f, r)
        piece = board.piece_at(sq)
        if piece is not None:
            if piece.color != color:
                orthogonal = (df == 0 or dr == 0)
                diagonal = (df != 0 and dr != 0)

                if orthogonal and piece.piece_type in (chess.ROOK, chess.QUEEN):
                    return sq
                if diagonal and piece.piece_type in (chess.BISHOP, chess.QUEEN):
                    return sq
            return None

        f += df
        r += dr

    return None


def generate_gold_annotations(fen: str) -> AnnotationBundle:
    board = chess.Board(fen)
    stm = board.turn
    opp = not stm

    legal_moves = list(board.legal_moves)

    arrows: List[ArrowAnnotation] = []
    arrow_seen: Set[Tuple[chess.Square, chess.Square, str]] = set()

    def add_arrow(from_sq: chess.Square, to_sq: chess.Square, kind: str, color: str) -> None:
        key = (from_sq, to_sq, kind)
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

    square_map: Dict[str, SquareAnnotation] = {}

    # 1) Tactical legal moves for side to move
    for move in legal_moves:
        is_capture = board.is_capture(move)
        gives_check = board.gives_check(move)

        if not is_capture and not gives_check:
            continue

        if is_capture and gives_check:
            kind = "capture_check"
            color = COLOR_CAPTURE_CHECK
        elif is_capture:
            kind = "capture"
            color = COLOR_CAPTURE
        else:
            kind = "check"
            color = COLOR_CHECK

        add_arrow(move.from_square, move.to_square, kind, color)

    # 2) Opponent pressure on side-to-move pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        if piece.color != stm:
            continue

        opp_attackers = sorted(board.attackers(opp, square))
        if not opp_attackers:
            continue

        sq_name = _square_name(square)
        square_map.setdefault(
            sq_name,
            SquareAnnotation(
                square=sq_name,
                kind="under_pressure",
                color=COLOR_OPP_PRESSURE_TARGET,
            ),
        )

        for attacker_sq in opp_attackers:
            add_arrow(attacker_sq, square, "opponent_pressure", COLOR_OPP_PRESSURE)

    # 3) Capture targets for side to move
    for move in legal_moves:
        if board.is_capture(move):
            sq = _square_name(move.to_square)
            square_map.setdefault(
                sq,
                SquareAnnotation(square=sq, kind="capture_target", color=COLOR_CAPTURE_TARGET),
            )

    # 4) Hanging enemy pieces for side to move
    for square in chess.SQUARES:
        if _is_hanging_piece(board, square, stm):
            sq = _square_name(square)
            square_map[sq] = SquareAnnotation(square=sq, kind="hanging_piece", color=COLOR_HANGING)

    # 5) Pinned pieces on side to move
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        if piece.color != stm:
            continue
        if piece.piece_type == chess.KING:
            continue

        if board.is_pinned(stm, square):
            sq_name = _square_name(square)
            square_map[sq_name] = SquareAnnotation(
                square=sq_name,
                kind="pinned_piece",
                color=COLOR_PINNED,
            )

            pinner_sq = _find_pinner(board, stm, square)
            if pinner_sq is not None:
                add_arrow(pinner_sq, square, "pin", COLOR_PIN_ARROW)

    # 6) Current check status
    if board.is_check():
        checked_king_square = board.king(board.turn)
        if checked_king_square is not None:
            sq = _square_name(checked_king_square)
            square_map[sq] = SquareAnnotation(square=sq, kind="checked_king", color=COLOR_CHECKED_KING)

        for checker_square in board.checkers():
            sq = _square_name(checker_square)
            if sq not in square_map:
                square_map[sq] = SquareAnnotation(square=sq, kind="checking_piece", color=COLOR_CHECKER)

    arrows.sort(key=lambda a: (chess.parse_square(a.from_square), chess.parse_square(a.to_square), a.kind))
    highlighted_squares = sorted(square_map.values(), key=lambda x: x.square)

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

    checked_square = board.king(board.turn) if board.is_check() else None

    svg = chess.svg.board(
        board=board,
        orientation=board_orientation,
        arrows=svg_arrows,
        fill=fill,
        check=checked_square,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic chess annotations from a FEN.")
    parser.add_argument("fen", help="FEN string")
    parser.add_argument("--png", default="annotated_board.png", help="Output PNG path")
    parser.add_argument("--json", default="annotations.json", help="Output JSON path")
    parser.add_argument("--size", type=int, default=720, help="PNG size in pixels")
    parser.add_argument(
        "--orientation",
        choices=["side_to_move", "white", "black"],
        default="side_to_move",
        help="Board orientation for rendering",
    )
    args = parser.parse_args()

    annotations = generate_gold_annotations(args.fen)
    save_annotations_json(annotations, args.json)
    render_annotations_to_png(args.fen, annotations, args.png, size=args.size, orientation=args.orientation)

    print(f"Saved JSON to {args.json}")
    print(f"Saved PNG to {args.png}")
    print(f"Arrows: {len(annotations.arrows)}")
    print(f"Highlighted squares: {len(annotations.highlighted_squares)}")


if __name__ == "__main__":
    main()