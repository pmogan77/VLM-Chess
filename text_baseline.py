#!/usr/bin/env python3
"""
Batch run text-only chess annotation from FEN.

Reads:
    puzzles/annotations_json/<id>.json
Each input JSON must contain a top level field:
    "fen"

Writes:
    puzzles/<model_name>-text-only/annotations_json/<id>.json
    puzzles/<model_name>-text-only/annotated_boards_debug/<id>.png
    puzzles/<model_name>-text-only/annotated_boards_final/<id>.png

This baseline:
- does NOT read the board image
- does NOT do a grounding stage
- sends only text (including FEN) to the model
- validates returned arrows against the true board from FEN
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import requests
import chess
import chess.svg
import cairosvg


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

TEXT_ONLY_SYSTEM_PROMPT = """
You are a chess text-only annotation assistant.

You will be given a legal chess FEN string that fully specifies the board position and side to move.

Return:
1. candidate_move_arrows: plausible candidate moves for the side to move
2. threat_arrows: opponent threats toward one of my pieces or an important threatened square
3. key_squares: important squares in the position

Rules:
- Return JSON only
- No markdown
- No commentary
- No SAN
- Do not repeat the FEN
- Do not reconstruct the board in prose
- No duplicate arrows
- No duplicate key squares
- Use algebraic squares like e4, f7, a1
- Use standard piece symbols: K Q R B N P for White, k q r b n p for Black
- Every candidate move arrow must start from a piece belonging to the side to move in the given FEN
- Every threat arrow must start from a piece belonging to the opponent side in the given FEN
- Candidate moves should be plausible moves for the side to move
- Threat arrows should point to one of my pieces or a clearly threatened square near my king or position
- If unsure, omit the arrow rather than guess
""".strip()


COLOR_CANDIDATE_MOVE = "#2e7d32"
COLOR_THREAT = "#fb8c00"
COLOR_INVALID_MOVE = "#9e9e9e"
COLOR_KEY_SQUARE = "#42a5f566"
COLOR_MY_PIECE = "#66bb6a66"
COLOR_OPP_PIECE = "#ef535066"


def parse_json_content(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        return json.loads(content)
    raise ValueError(f"Unexpected message content type: {type(content)}")


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def piece_record_key(item: Dict[str, str]) -> str:
    return f'{item.get("square","")}::{item.get("piece","")}'


def dedupe_piece_records(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for item in items:
        key = piece_record_key(item)
        if key not in seen:
            seen.add(key)
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


def build_text_only_schema(
    max_candidate_arrows: int,
    max_threat_arrows: int,
    max_key_squares: int,
) -> Dict[str, Any]:
    square_pattern = "^[a-h][1-8]$"
    piece_pattern = "^[KQRBNPkqrbnp]$"

    arrow_obj = {
        "type": "object",
        "properties": {
            "from": {"type": "string", "pattern": square_pattern},
            "to": {"type": "string", "pattern": square_pattern},
            "piece": {"type": "string", "pattern": piece_pattern},
        },
        "required": ["from", "to", "piece"],
        "additionalProperties": False,
    }

    return {
        "name": "chess_text_only_arrows",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "side_to_move": {"type": "string", "enum": ["white", "black"]},
                "candidate_move_arrows": {
                    "type": "array",
                    "items": arrow_obj,
                    "maxItems": max_candidate_arrows,
                },
                "threat_arrows": {
                    "type": "array",
                    "items": arrow_obj,
                    "maxItems": max_threat_arrows,
                },
                "key_squares": {
                    "type": "array",
                    "items": {"type": "string", "pattern": square_pattern},
                    "maxItems": max_key_squares,
                },
            },
            "required": ["side_to_move", "candidate_move_arrows", "threat_arrows", "key_squares"],
            "additionalProperties": False,
        },
    }


def build_text_only_user_prompt(
    fen: str,
    side_to_move: str,
    max_candidate_arrows: int,
    max_threat_arrows: int,
    max_key_squares: int,
) -> str:
    return (
        f"The full chess position is given by this FEN: {fen} "
        f"The side to move is {side_to_move}. "
        f"Using only this FEN position, return up to {max_candidate_arrows} candidate move arrows, "
        f"up to {max_threat_arrows} threat arrows, and up to {max_key_squares} key squares. "
        f"Each candidate move arrow must start from one of my pieces in the FEN and use the correct piece symbol. "
        f"Each threat arrow must start from one of the opponent pieces in the FEN and use the correct piece symbol. "
        f"Threat arrows should point to one of my pieces or an important threatened square near my king or position. "
        f"Return JSON only matching the schema."
    )


def call_openrouter(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Chess Text Only FEN Arrow Generator",
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    if "error" in data:
        raise RuntimeError(json.dumps(data["error"], indent=2))
    return data


def request_text_only_arrows(
    api_key: str,
    fen: str,
    side_to_move: str,
    model: str,
    max_candidate_arrows: int,
    max_threat_arrows: int,
    max_key_squares: int,
    reasoning_enabled: bool,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TEXT_ONLY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_text_only_user_prompt(
                    fen=fen,
                    side_to_move=side_to_move,
                    max_candidate_arrows=max_candidate_arrows,
                    max_threat_arrows=max_threat_arrows,
                    max_key_squares=max_key_squares,
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": build_text_only_schema(
                max_candidate_arrows=max_candidate_arrows,
                max_threat_arrows=max_threat_arrows,
                max_key_squares=max_key_squares,
            ),
        },
        "plugins": [{"id": "response-healing"}],
        "stream": False,
    }

    if reasoning_enabled:
        payload["reasoning"] = {"enabled": True}

    data = call_openrouter(api_key, payload)
    message = data["choices"][0]["message"]
    parsed = parse_json_content(message["content"])
    return {"parsed_json": parsed, "raw_response": data}


def piece_list_for_color(board: chess.Board, color: chess.Color) -> List[Dict[str, str]]:
    out = []
    for sq, piece in sorted(board.piece_map().items()):
        if piece.color == color:
            out.append(
                {
                    "square": chess.square_name(sq),
                    "piece": piece.symbol(),
                }
            )
    return out


def piece_lookup(items: List[Dict[str, str]]) -> Dict[str, str]:
    return {item["square"]: item["piece"] for item in items}


def is_slider_clear(board: chess.Board, from_square: int, to_square: int) -> bool:
    file_diff = chess.square_file(to_square) - chess.square_file(from_square)
    rank_diff = chess.square_rank(to_square) - chess.square_rank(from_square)

    step_file = 0 if file_diff == 0 else (1 if file_diff > 0 else -1)
    step_rank = 0 if rank_diff == 0 else (1 if rank_diff > 0 else -1)

    cur_file = chess.square_file(from_square) + step_file
    cur_rank = chess.square_rank(from_square) + step_rank

    while (cur_file, cur_rank) != (chess.square_file(to_square), chess.square_rank(to_square)):
        sq = chess.square(cur_file, cur_rank)
        if board.piece_at(sq) is not None:
            return False
        cur_file += step_file
        cur_rank += step_rank

    return True


def is_plausible_threat_arrow(board: chess.Board, from_square: int, to_square: int, piece_symbol: str) -> bool:
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
        return abs_file == abs_rank and abs_file != 0 and is_slider_clear(board, from_square, to_square)
    if piece_type == chess.ROOK:
        return ((file_diff == 0) != (rank_diff == 0)) and is_slider_clear(board, from_square, to_square)
    if piece_type == chess.QUEEN:
        straight = ((file_diff == 0) != (rank_diff == 0))
        diagonal = abs_file == abs_rank and abs_file != 0
        return (straight or diagonal) and is_slider_clear(board, from_square, to_square)
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
        record = {"from": from_sq, "to": to_sq, "piece": piece_symbol}

        if from_sq not in my_lookup:
            invalid.append({**record, "reason": "from_not_in_my_piece_set"})
            continue

        if my_lookup[from_sq] != piece_symbol:
            invalid.append({**record, "reason": "piece_mismatch_with_board"})
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

    return {"valid": valid, "invalid": invalid}


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
        record = {"from": from_sq, "to": to_sq, "piece": piece_symbol}

        if from_sq not in opp_lookup:
            invalid.append({**record, "reason": "from_not_in_opponent_piece_set"})
            continue

        if opp_lookup[from_sq] != piece_symbol:
            invalid.append({**record, "reason": "piece_mismatch_with_board"})
            continue

        try:
            from_square = chess.parse_square(from_sq)
            to_square = chess.parse_square(to_sq)
        except ValueError:
            invalid.append({**record, "reason": "invalid_square"})
            continue

        if not is_plausible_threat_arrow(board, from_square, to_square, piece_symbol):
            invalid.append({**record, "reason": "not_a_plausible_threat_geometry"})
            continue

        if to_sq in my_squares:
            valid.append(record)
        else:
            valid.append({**record, "reason": "threatened_square_not_my_piece"})

    return {"valid": valid, "invalid": invalid}


def validate_key_squares(items: List[str]) -> List[str]:
    valid = []
    for sq in dedupe_keep_order(items):
        try:
            chess.parse_square(sq)
            valid.append(sq)
        except ValueError:
            continue
    return valid


def render_output(
    fen: str,
    side_to_move: str,
    my_valid_pieces: List[Dict[str, str]],
    opponent_valid_pieces: List[Dict[str, str]],
    candidate_valid: List[Dict[str, str]],
    candidate_all: List[Dict[str, str]],
    threat_valid: List[Dict[str, str]],
    threat_all: List[Dict[str, str]],
    key_squares: List[str],
    output_png: str,
    size: int,
    orientation_mode: str,
    mode: str,
) -> None:
    board = chess.Board(fen)

    if orientation_mode == "side_to_move":
        orientation = chess.WHITE if side_to_move == "white" else chess.BLACK
    elif orientation_mode == "white":
        orientation = chess.WHITE
    else:
        orientation = chess.BLACK

    arrows = []
    fill = {}

    if mode == "debug":
        for item in my_valid_pieces:
            fill[chess.parse_square(item["square"])] = COLOR_MY_PIECE

        for item in opponent_valid_pieces:
            sq = chess.parse_square(item["square"])
            if sq not in fill:
                fill[sq] = COLOR_OPP_PIECE

        for sq in key_squares:
            fill[chess.parse_square(sq)] = COLOR_KEY_SQUARE

    elif mode == "final":
        for sq in key_squares:
            fill[chess.parse_square(sq)] = COLOR_KEY_SQUARE

    valid_candidate_keys = {arrow_record_key(x) for x in candidate_valid}
    valid_threat_keys = {arrow_record_key(x) for x in threat_valid}

    for item in dedupe_arrow_records(candidate_all):
        try:
            tail = chess.parse_square(item["from"])
            head = chess.parse_square(item["to"])
        except Exception:
            continue
        key = arrow_record_key(item)
        if key in valid_candidate_keys:
            arrows.append(chess.svg.Arrow(tail=tail, head=head, color=COLOR_CANDIDATE_MOVE))
        elif mode == "debug":
            arrows.append(chess.svg.Arrow(tail=tail, head=head, color=COLOR_INVALID_MOVE))

    for item in dedupe_arrow_records(threat_all):
        try:
            tail = chess.parse_square(item["from"])
            head = chess.parse_square(item["to"])
        except Exception:
            continue
        key = arrow_record_key(item)
        if key in valid_threat_keys:
            arrows.append(chess.svg.Arrow(tail=tail, head=head, color=COLOR_THREAT))
        elif mode == "debug":
            arrows.append(chess.svg.Arrow(tail=tail, head=head, color=COLOR_INVALID_MOVE))

    svg_data = chess.svg.board(
        board=board,
        size=size,
        orientation=orientation,
        arrows=arrows,
        fill=fill,
        coordinates=True,
    )
    cairosvg.svg2png(bytestring=svg_data.encode("utf-8"), write_to=output_png)


def sanitize_model_name(model: str) -> str:
    name = model.strip().split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    return name or "model"


def output_set_is_complete(
    output_annotation_json_path: Path,
    output_debug_png_path: Path,
    output_final_png_path: Path,
) -> bool:
    if not output_annotation_json_path.exists():
        return False
    if not output_debug_png_path.exists():
        return False
    if not output_final_png_path.exists():
        return False

    try:
        with open(output_annotation_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = [
            "fen",
            "model",
            "valid_my_pieces",
            "valid_opponent_pieces",
            "valid_candidate_move_arrows",
            "valid_threat_arrows",
            "key_squares",
            "debug_png",
            "final_png",
        ]
        return all(key in data for key in required_keys)
    except Exception:
        return False


def process_one(
    input_annotation_json_path: Path,
    output_annotation_json_path: Path,
    output_debug_png_path: Path,
    output_final_png_path: Path,
    api_key: str,
    model: str,
    max_candidate_arrows: int,
    max_threat_arrows: int,
    max_key_squares: int,
    size: int,
    orientation: str,
    reasoning_enabled: bool,
) -> Dict[str, Any]:
    with open(input_annotation_json_path, "r", encoding="utf-8") as f:
        input_meta = json.load(f)

    fen = input_meta.get("fen")
    if not fen:
        raise ValueError(f"Missing top-level 'fen' in {input_annotation_json_path}")

    board = chess.Board(fen)
    side_to_move = "white" if board.turn == chess.WHITE else "black"

    my_pieces = dedupe_piece_records(piece_list_for_color(board, board.turn))
    opponent_pieces = dedupe_piece_records(piece_list_for_color(board, not board.turn))

    result = request_text_only_arrows(
        api_key=api_key,
        fen=fen,
        side_to_move=side_to_move,
        model=model,
        max_candidate_arrows=max_candidate_arrows,
        max_threat_arrows=max_threat_arrows,
        max_key_squares=max_key_squares,
        reasoning_enabled=reasoning_enabled,
    )

    candidate_move_arrows = dedupe_arrow_records(result["parsed_json"].get("candidate_move_arrows", []))
    threat_arrows = dedupe_arrow_records(result["parsed_json"].get("threat_arrows", []))
    key_squares = validate_key_squares(result["parsed_json"].get("key_squares", []))

    candidate_validation = validate_candidate_arrows(
        board=board,
        my_valid_pieces=my_pieces,
        items=candidate_move_arrows,
    )

    threat_validation = validate_threat_arrows(
        board=board,
        opponent_valid_pieces=opponent_pieces,
        my_valid_pieces=my_pieces,
        items=threat_arrows,
    )

    render_output(
        fen=fen,
        side_to_move=side_to_move,
        my_valid_pieces=my_pieces,
        opponent_valid_pieces=opponent_pieces,
        candidate_valid=candidate_validation["valid"],
        candidate_all=candidate_move_arrows,
        threat_valid=threat_validation["valid"],
        threat_all=threat_arrows,
        key_squares=key_squares,
        output_png=str(output_debug_png_path),
        size=size,
        orientation_mode=orientation,
        mode="debug",
    )

    render_output(
        fen=fen,
        side_to_move=side_to_move,
        my_valid_pieces=my_pieces,
        opponent_valid_pieces=opponent_pieces,
        candidate_valid=candidate_validation["valid"],
        candidate_all=candidate_move_arrows,
        threat_valid=threat_validation["valid"],
        threat_all=threat_arrows,
        key_squares=key_squares,
        output_png=str(output_final_png_path),
        size=size,
        orientation_mode=orientation,
        mode="final",
    )

    parsed_output = {
        "id": input_annotation_json_path.stem,
        "input_annotation_json_path": str(input_annotation_json_path),
        "side_to_move": side_to_move,
        "fen": fen,
        "model": model,
        "baseline_type": "text_only_fen_baseline",
        "my_pieces": my_pieces,
        "opponent_pieces": opponent_pieces,
        "valid_my_pieces": my_pieces,
        "invalid_my_pieces": [],
        "valid_opponent_pieces": opponent_pieces,
        "invalid_opponent_pieces": [],
        "candidate_move_arrows": candidate_move_arrows,
        "valid_candidate_move_arrows": candidate_validation["valid"],
        "invalid_candidate_move_arrows": candidate_validation["invalid"],
        "threat_arrows": threat_arrows,
        "valid_threat_arrows": threat_validation["valid"],
        "invalid_threat_arrows": threat_validation["invalid"],
        "key_squares": key_squares,
        "debug_png": str(output_debug_png_path),
        "final_png": str(output_final_png_path),
    }

    with open(output_annotation_json_path, "w", encoding="utf-8") as f:
        json.dump(parsed_output, f, indent=2)

    return parsed_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch run text-only FEN chess annotation over puzzles/annotations_json."
    )
    parser.add_argument("--puzzles-dir", required=True, help="Path to the root puzzles directory.")
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"), help="OpenRouter API key.")
    parser.add_argument("--model", default="google/gemma-4-31b-it", help="OpenRouter model.")
    parser.add_argument("--max-candidate-arrows", type=int, default=2)
    parser.add_argument("--max-threat-arrows", type=int, default=3)
    parser.add_argument("--max-key-squares", type=int, default=3)
    parser.add_argument("--size", type=int, default=720)
    parser.add_argument("--orientation", choices=["side_to_move", "white", "black"], default="side_to_move")
    parser.add_argument("--no-reasoning", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing OpenRouter API key. Pass --api-key or set OPENROUTER_API_KEY.")

    puzzles_dir = Path(args.puzzles_dir)
    input_annotations_dir = puzzles_dir / "annotations_json"

    if not input_annotations_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {input_annotations_dir}")

    model_dir = puzzles_dir / f"{sanitize_model_name(args.model)}-text-only"
    output_annotations_dir = model_dir / "annotations_json"
    output_debug_dir = model_dir / "annotated_boards_debug"
    output_final_dir = model_dir / "annotated_boards_final"

    output_annotations_dir.mkdir(parents=True, exist_ok=True)
    output_debug_dir.mkdir(parents=True, exist_ok=True)
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
        output_debug_png_path = output_debug_dir / f"{stem}.png"
        output_final_png_path = output_final_dir / f"{stem}.png"

        if (not args.force) and output_set_is_complete(
            output_annotation_json_path=output_annotation_json_path,
            output_debug_png_path=output_debug_png_path,
            output_final_png_path=output_final_png_path,
        ):
            print(json.dumps({
                "id": stem,
                "status": "skipped_complete",
                "debug_png": str(output_debug_png_path),
                "final_png": str(output_final_png_path),
                "json": str(output_annotation_json_path),
            }))
            continue

        try:
            result = process_one(
                input_annotation_json_path=input_annotation_json_path,
                output_annotation_json_path=output_annotation_json_path,
                output_debug_png_path=output_debug_png_path,
                output_final_png_path=output_final_png_path,
                api_key=args.api_key,
                model=args.model,
                max_candidate_arrows=args.max_candidate_arrows,
                max_threat_arrows=args.max_threat_arrows,
                max_key_squares=args.max_key_squares,
                size=args.size,
                orientation=args.orientation,
                reasoning_enabled=not args.no_reasoning,
            )
            processed += 1
            print(json.dumps({
                "id": stem,
                "status": "ok",
                "debug_png": str(output_debug_png_path),
                "final_png": str(output_final_png_path),
                "json": str(output_annotation_json_path),
                "valid_candidate_move_arrows": len(result["valid_candidate_move_arrows"]),
                "valid_threat_arrows": len(result["valid_threat_arrows"]),
            }))
        except Exception as e:
            errors.append({"id": stem, "error": str(e)})
            print(json.dumps({"id": stem, "status": "error", "error": str(e)}))

    summary = {
        "puzzles_dir": str(puzzles_dir),
        "model_output_dir": str(model_dir),
        "processed": processed,
        "errors": len(errors),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

# TODO: do not validate here, do the candidate generation as a followup to this conversation