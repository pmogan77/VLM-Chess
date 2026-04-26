import json
from pathlib import Path
from typing import Dict, List, Any

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


COLOR_INVALID_MOVE = "#9e9e9e"


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def arrow_record_key(item: Dict[str, str]) -> str:
    return f'{item.get("from","")}::{item.get("to","")}::{item.get("piece","")}'


def dedupe_arrow_records(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for item in items:
        key = arrow_record_key(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def dedupe_piece_records(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for item in items:
        key = f'{item.get("square","")}::{item.get("piece","")}'
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


def piece_lookup(items: List[Dict[str, str]]) -> Dict[str, str]:
    return {item["square"]: item["piece"] for item in items}


def is_slider_clear(board: chess.Board, from_square: int, to_square: int) -> bool:
    file_diff = chess.square_file(to_square) - chess.square_file(from_square)
    rank_diff = chess.square_rank(to_square) - chess.square_rank(from_square)

    step_file = 0 if file_diff == 0 else (1 if file_diff > 0 else -1)
    step_rank = 0 if rank_diff == 0 else (1 if rank_diff > 0 else -1)

    cur_file = chess.square_file(from_square) + step_file
    cur_rank = chess.square_rank(from_square) + step_rank

    while (cur_file, cur_rank) != (
        chess.square_file(to_square),
        chess.square_rank(to_square),
    ):
        sq = chess.square(cur_file, cur_rank)
        if board.piece_at(sq) is not None:
            return False
        cur_file += step_file
        cur_rank += step_rank

    return True


def is_plausible_threat_arrow(
    board: chess.Board,
    from_square: int,
    to_square: int,
    piece_symbol: str,
) -> bool:
    board_piece = board.piece_at(from_square)

    if board_piece is None or board_piece.symbol() != piece_symbol:
        return False

    piece_type = board_piece.piece_type

    file_diff = chess.square_file(to_square) - chess.square_file(from_square)
    rank_diff = chess.square_rank(to_square) - chess.square_rank(from_square)

    abs_file = abs(file_diff)
    abs_rank = abs(rank_diff)

    if piece_type == chess.KNIGHT:
        return (abs_file, abs_rank) in {(1, 2), (2, 1)}

    if piece_type == chess.BISHOP:
        return (
            abs_file == abs_rank
            and abs_file != 0
            and is_slider_clear(board, from_square, to_square)
        )

    if piece_type == chess.ROOK:
        return (
            ((file_diff == 0) != (rank_diff == 0))
            and is_slider_clear(board, from_square, to_square)
        )

    if piece_type == chess.QUEEN:
        straight = ((file_diff == 0) != (rank_diff == 0))
        diagonal = abs_file == abs_rank and abs_file != 0

        return (
            (straight or diagonal)
            and is_slider_clear(board, from_square, to_square)
        )

    if piece_type == chess.KING:
        return max(abs_file, abs_rank) == 1

    if piece_type == chess.PAWN:
        if board_piece.color == chess.WHITE:
            return rank_diff == 1 and abs_file == 1

        return rank_diff == -1 and abs_file == 1

    return False


def validate_candidate_arrows(
    board: chess.Board,
    my_valid_pieces: List[Dict[str, str]],
    items: List[Dict[str, str]],
) -> Dict[str, List[Dict[str, str]]]:
    my_lookup = piece_lookup(my_valid_pieces)

    valid = []
    invalid = []

    for item in dedupe_arrow_records(items):
        from_sq = item.get("from", "")
        to_sq = item.get("to", "")
        piece_symbol = item.get("piece", "")

        record = {
            "from": from_sq,
            "to": to_sq,
            "piece": piece_symbol,
        }

        if from_sq not in my_lookup:
            invalid.append({**record, "reason": "from_not_in_my_grounded_list"})
            continue

        if my_lookup[from_sq] != piece_symbol:
            invalid.append({**record, "reason": "piece_mismatch_with_grounded_list"})
            continue

        try:
            from_square = chess.parse_square(from_sq)
            to_square = chess.parse_square(to_sq)
        except ValueError:
            invalid.append({**record, "reason": "invalid_square"})
            continue

        move = chess.Move(from_square, to_square)

        if move in board.legal_moves:
            valid.append(record)
        else:
            invalid.append({**record, "reason": "illegal_move"})

    return {
        "valid": valid,
        "invalid": invalid,
    }


def validate_threat_arrows(
    board: chess.Board,
    opponent_valid_pieces: List[Dict[str, str]],
    my_valid_pieces: List[Dict[str, str]],
    items: List[Dict[str, str]],
) -> Dict[str, List[Dict[str, str]]]:
    opp_lookup = piece_lookup(opponent_valid_pieces)
    my_squares = {item["square"] for item in my_valid_pieces}

    valid = []
    invalid = []

    for item in dedupe_arrow_records(items):
        from_sq = item.get("from", "")
        to_sq = item.get("to", "")
        piece_symbol = item.get("piece", "")

        record = {
            "from": from_sq,
            "to": to_sq,
            "piece": piece_symbol,
        }

        if from_sq not in opp_lookup:
            invalid.append({**record, "reason": "from_not_in_opponent_grounded_list"})
            continue

        if opp_lookup[from_sq] != piece_symbol:
            invalid.append({**record, "reason": "piece_mismatch_with_grounded_list"})
            continue

        try:
            from_square = chess.parse_square(from_sq)
            to_square = chess.parse_square(to_sq)
        except ValueError:
            invalid.append({**record, "reason": "invalid_square"})
            continue

        if not is_plausible_threat_arrow(
            board=board,
            from_square=from_square,
            to_square=to_square,
            piece_symbol=piece_symbol,
        ):
            invalid.append({**record, "reason": "not_a_plausible_threat_geometry"})
            continue

        if to_sq in my_squares:
            valid.append(record)
        else:
            valid.append({**record, "reason": "threatened_square_not_my_piece"})

    return {
        "valid": valid,
        "invalid": invalid,
    }


def build_annotation_bundle(
    fen: str,
    side_to_move: str,
    candidate_valid: List[Dict[str, str]],
    candidate_all: List[Dict[str, str]],
    threat_valid: List[Dict[str, str]],
    threat_all: List[Dict[str, str]],
    key_squares: List[str],
    mode: str,
) -> AnnotationBundle:
    """
    Convert text-only JSON arrows into the shared gold AnnotationBundle.

    mode="final":
        Only valid arrows are rendered.

    mode="debug":
        Valid arrows are rendered in normal colors.
        Invalid arrows are rendered in gray.
    """

    arrows: List[ArrowAnnotation] = []
    highlighted_squares: List[SquareAnnotation] = []

    valid_candidate_keys = {arrow_record_key(x) for x in candidate_valid}
    valid_threat_keys = {arrow_record_key(x) for x in threat_valid}

    for sq in validate_key_squares(key_squares):
        highlighted_squares.append(
            SquareAnnotation(
                square=sq,
                kind="key_square",
                color=COLOR_KEY_SQUARE,
            )
        )

    for item in dedupe_arrow_records(candidate_all):
        try:
            chess.parse_square(item["from"])
            chess.parse_square(item["to"])
        except Exception:
            continue

        key = arrow_record_key(item)

        if key in valid_candidate_keys:
            color = COLOR_CANDIDATE_MOVE
            kind = "candidate_move"
        elif mode == "debug":
            color = COLOR_INVALID_MOVE
            kind = "invalid_candidate_move"
        else:
            continue

        arrows.append(
            ArrowAnnotation(
                from_square=item["from"],
                to_square=item["to"],
                kind=kind,
                color=color,
            )
        )

    for item in dedupe_arrow_records(threat_all):
        try:
            chess.parse_square(item["from"])
            chess.parse_square(item["to"])
        except Exception:
            continue

        key = arrow_record_key(item)

        if key in valid_threat_keys:
            color = COLOR_THREAT
            kind = "threat"
        elif mode == "debug":
            color = COLOR_INVALID_MOVE
            kind = "invalid_threat"
        else:
            continue

        arrows.append(
            ArrowAnnotation(
                from_square=item["from"],
                to_square=item["to"],
                kind=kind,
                color=color,
            )
        )

    return AnnotationBundle(
        fen=fen,
        side_to_move=side_to_move,
        arrows=arrows,
        highlighted_squares=highlighted_squares,
    )


def render_text_only_output(
    fen: str,
    side_to_move: str,
    candidate_valid: List[Dict[str, str]],
    candidate_all: List[Dict[str, str]],
    threat_valid: List[Dict[str, str]],
    threat_all: List[Dict[str, str]],
    key_squares: List[str],
    output_png: str,
    size: int,
    orientation: str,
    mode: str,
) -> None:
    annotations = build_annotation_bundle(
        fen=fen,
        side_to_move=side_to_move,
        candidate_valid=candidate_valid,
        candidate_all=candidate_all,
        threat_valid=threat_valid,
        threat_all=threat_all,
        key_squares=key_squares,
        mode=mode,
    )

    render_annotations_to_png(
        fen=fen,
        annotations=annotations,
        output_png=output_png,
        size=size,
        orientation=orientation,
    )


def process_one_json(
    json_path: Path,
    debug_dir: Path,
    final_dir: Path,
    size: int = 720,
    orientation: str = "side_to_move",
) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fen = data["fen"]
    board = chess.Board(fen)

    side_to_move = data.get("side_to_move")

    if side_to_move not in {"white", "black"}:
        side_to_move = "white" if board.turn == chess.WHITE else "black"

    ann = data.get("annotation_data", {})

    candidate_move_arrows = dedupe_arrow_records(
        ann.get("candidate_move_arrows", [])
    )

    threat_arrows = dedupe_arrow_records(
        ann.get("threat_arrows", [])
    )

    key_squares = validate_key_squares(
        ann.get("key_squares", [])
    )

    my_valid_pieces = []
    opponent_valid_pieces = []

    my_color = chess.WHITE if side_to_move == "white" else chess.BLACK

    for square, piece in board.piece_map().items():
        record = {
            "square": chess.square_name(square),
            "piece": piece.symbol(),
        }

        if piece.color == my_color:
            my_valid_pieces.append(record)
        else:
            opponent_valid_pieces.append(record)

    my_valid_pieces = dedupe_piece_records(my_valid_pieces)
    opponent_valid_pieces = dedupe_piece_records(opponent_valid_pieces)

    candidate_validation = validate_candidate_arrows(
        board=board,
        my_valid_pieces=my_valid_pieces,
        items=candidate_move_arrows,
    )

    threat_validation = validate_threat_arrows(
        board=board,
        opponent_valid_pieces=opponent_valid_pieces,
        my_valid_pieces=my_valid_pieces,
        items=threat_arrows,
    )

    debug_png = debug_dir / f"{json_path.stem}.png"
    final_png = final_dir / f"{json_path.stem}.png"

    render_text_only_output(
        fen=fen,
        side_to_move=side_to_move,
        candidate_valid=candidate_validation["valid"],
        candidate_all=candidate_move_arrows,
        threat_valid=threat_validation["valid"],
        threat_all=threat_arrows,
        key_squares=key_squares,
        output_png=str(debug_png),
        size=size,
        orientation=orientation,
        mode="debug",
    )

    render_text_only_output(
        fen=fen,
        side_to_move=side_to_move,
        candidate_valid=candidate_validation["valid"],
        candidate_all=candidate_move_arrows,
        threat_valid=threat_validation["valid"],
        threat_all=threat_arrows,
        key_squares=key_squares,
        output_png=str(final_png),
        size=size,
        orientation=orientation,
        mode="final",
    )

    return {
        "id": data.get("id", json_path.stem),
        "json_path": str(json_path),
        "debug_png": str(debug_png),
        "final_png": str(final_png),
        "valid_candidate_move_arrows": candidate_validation["valid"],
        "invalid_candidate_move_arrows": candidate_validation["invalid"],
        "valid_threat_arrows": threat_validation["valid"],
        "invalid_threat_arrows": threat_validation["invalid"],
    }


def process_all(
    model_dir: str,
    size: int = 720,
    orientation: str = "side_to_move",
) -> None:
    model_path = Path(model_dir)

    text_only_dir = model_path / "text_only"
    debug_dir = text_only_dir / "debug"
    final_dir = text_only_dir / "final"

    debug_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted((text_only_dir / "moves").glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {text_only_dir}")

    processed = 0
    errors = []

    for json_path in json_files:
        try:
            result = process_one_json(
                json_path=json_path,
                debug_dir=debug_dir,
                final_dir=final_dir,
                size=size,
                orientation=orientation,
            )

            processed += 1

            print(json.dumps({
                "id": result["id"],
                "status": "ok",
                "debug_png": result["debug_png"],
                "final_png": result["final_png"],
                "valid_candidate_count": len(result["valid_candidate_move_arrows"]),
                "invalid_candidate_count": len(result["invalid_candidate_move_arrows"]),
                "valid_threat_count": len(result["valid_threat_arrows"]),
                "invalid_threat_count": len(result["invalid_threat_arrows"]),
            }))

        except Exception as e:
            errors.append({
                "id": json_path.stem,
                "error": str(e),
            })

            print(json.dumps({
                "id": json_path.stem,
                "status": "error",
                "error": str(e),
            }))

    print(json.dumps({
        "model_dir": str(model_path),
        "text_only_dir": str(text_only_dir),
        "processed": processed,
        "errors": len(errors),
    }, indent=2))


if __name__ == "__main__":
    process_all("puzzles/gemma-4-31b-it")