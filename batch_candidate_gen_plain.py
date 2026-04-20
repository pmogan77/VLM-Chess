#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List

import requests
import chess


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """
You are a chess vision assistant.
Given a chessboard image and the side to move, return a short list of strong candidate moves.

Rules:
- Return UCI moves only
- Return JSON only
- No markdown
- No commentary
- No SAN
- No duplicate moves
- Only include moves for the given side to move
- Prefer legal-looking, high-quality candidate moves
""".strip()


def image_to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def build_schema(max_candidates: int) -> Dict[str, Any]:
    return {
        "name": "chess_candidate_moves",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "candidate_moves": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "pattern": "^[a-h][1-8][a-h][1-8][qrbn]?$",
                    },
                    "minItems": 1,
                    "maxItems": max_candidates,
                }
            },
            "required": ["candidate_moves"],
            "additionalProperties": False,
        },
    }


def build_user_prompt(side_to_move: str, max_candidates: int) -> str:
    return (
        f"The side to move is {side_to_move}. "
        f"Look at the chessboard image and return exactly {max_candidates} candidate moves "
        f"in UCI notation as JSON matching the schema."
    )


def parse_json_content(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        return json.loads(content)
    raise ValueError(f"Unexpected message content type: {type(content)}")


def call_openrouter(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Chess Candidate Move Generator Plain",
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


def request_candidate_moves(
    api_key: str,
    image_path: str,
    side_to_move: str,
    model: str,
    max_candidates: int,
    reasoning_enabled: bool,
) -> Dict[str, Any]:
    image_data_url = image_to_data_url(image_path)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_user_prompt(side_to_move, max_candidates)},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": build_schema(max_candidates),
        },
        "plugins": [{"id": "response-healing"}],
        "stream": False,
    }

    if reasoning_enabled:
        payload["reasoning"] = {"enabled": True}

    data = call_openrouter(api_key, payload)
    message = data["choices"][0]["message"]
    parsed = parse_json_content(message["content"])

    candidate_moves = parsed.get("candidate_moves", [])
    if not isinstance(candidate_moves, list) or not candidate_moves:
        raise ValueError("No candidate_moves found in model response.")

    return {
        "model": model,
        "image_path": image_path,
        "side_to_move": side_to_move,
        "candidate_moves": candidate_moves,
        "raw_response": data,
    }


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def validate_candidate_moves(board: chess.Board, moves: List[str]) -> Dict[str, Any]:
    valid: List[str] = []
    invalid: List[Dict[str, str]] = []

    for move_text in dedupe_keep_order(moves):
        try:
            move = chess.Move.from_uci(move_text)
        except ValueError:
            invalid.append({"move": move_text, "reason": "invalid_uci"})
            continue

        if move in board.legal_moves:
            valid.append(move_text)
        else:
            invalid.append({"move": move_text, "reason": "illegal_move"})

    return {"valid": valid, "invalid": invalid}


def load_fen_and_side(meta_json_path: Path) -> Dict[str, str]:
    with open(meta_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fen = data.get("fen")
    if not fen:
        raise ValueError(f"Missing top-level 'fen' in {meta_json_path}")

    board = chess.Board(fen)
    side_to_move = "white" if board.turn == chess.WHITE else "black"
    return {"fen": fen, "side_to_move": side_to_move}


def output_set_is_complete(output_json_path: Path) -> bool:
    if not output_json_path.exists():
        return False

    try:
        with open(output_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = [
            "id",
            "image_path",
            "fen",
            "side_to_move",
            "model",
            "candidate_moves",
            "valid_candidate_moves",
            "invalid_candidate_moves",
        ]
        return all(key in data for key in required_keys)
    except Exception:
        return False


def process_one(
    image_path: Path,
    meta_json_path: Path,
    output_json_path: Path,
    raw_output_path: Path | None,
    api_key: str,
    model: str,
    max_candidates: int,
    reasoning_enabled: bool,
) -> Dict[str, Any]:
    meta = load_fen_and_side(meta_json_path)

    result = request_candidate_moves(
        api_key=api_key,
        image_path=str(image_path),
        side_to_move=meta["side_to_move"],
        model=model,
        max_candidates=max_candidates,
        reasoning_enabled=reasoning_enabled,
    )

    board = chess.Board(meta["fen"])
    validation = validate_candidate_moves(board, result["candidate_moves"])

    parsed_output = {
        "id": image_path.stem,
        "image_path": str(image_path),
        "input_annotation_json_path": str(meta_json_path),
        "fen": meta["fen"],
        "side_to_move": meta["side_to_move"],
        "model": model,
        "input_mode": "plain_image",
        "candidate_moves": result["candidate_moves"],
        "valid_candidate_moves": validation["valid"],
        "invalid_candidate_moves": validation["invalid"],
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(parsed_output, f, indent=2)

    if raw_output_path is not None:
        with open(raw_output_path, "w", encoding="utf-8") as f:
            json.dump(result["raw_response"], f, indent=2)

    return parsed_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch candidate generation from plain chessboard images."
    )
    parser.add_argument("--puzzles-dir", required=True, help="Root puzzles directory.")
    parser.add_argument("--model-dir-name", required=True, help="Folder name under puzzles/, for example gemma-31b.")
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"), help="OpenRouter API key.")
    parser.add_argument("--model", default="google/gemma-4-31b-it", help="Vision-capable OpenRouter model.")
    parser.add_argument("--num-candidates", type=int, default=5, help="How many candidate moves to request.")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning mode.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of boards to process.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N sorted boards.")
    parser.add_argument("--force", action="store_true", help="Reprocess even if JSON output already exists.")
    parser.add_argument("--save-raw", action="store_true", help="Also save raw OpenRouter responses.")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing OpenRouter API key. Pass --api-key or set OPENROUTER_API_KEY.")

    puzzles_dir = Path(args.puzzles_dir)
    model_dir = puzzles_dir / args.model_dir_name
    image_dir = puzzles_dir / "plain_boards"
    metadata_json_dir = puzzles_dir / "annotations_json"
    output_dir = model_dir / "candidate_gen_no_annotations"
    raw_dir = model_dir / "candidate_gen_no_annotations_raw"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {image_dir}")
    if not metadata_json_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {metadata_json_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.png"))
    if args.start_index:
        image_paths = image_paths[args.start_index:]
    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    processed = 0
    errors = []

    for image_path in image_paths:
        stem = image_path.stem
        meta_json_path = metadata_json_dir / f"{stem}.json"
        output_json_path = output_dir / f"{stem}.json"
        raw_output_path = (raw_dir / f"{stem}.json") if args.save_raw else None

        if not meta_json_path.exists():
            errors.append({"id": stem, "error": f"Missing metadata JSON: {meta_json_path}"})
            print(json.dumps({
                "id": stem,
                "status": "error",
                "error": f"missing_metadata_json:{meta_json_path}",
            }))
            continue

        if (not args.force) and output_set_is_complete(output_json_path):
            print(json.dumps({
                "id": stem,
                "status": "skipped_complete",
                "json": str(output_json_path),
            }))
            continue

        try:
            result = process_one(
                image_path=image_path,
                meta_json_path=meta_json_path,
                output_json_path=output_json_path,
                raw_output_path=raw_output_path,
                api_key=args.api_key,
                model=args.model,
                max_candidates=args.num_candidates,
                reasoning_enabled=not args.no_reasoning,
            )
            processed += 1
            print(json.dumps({
                "id": stem,
                "status": "ok",
                "json": str(output_json_path),
                "valid_candidate_moves": len(result["valid_candidate_moves"]),
                "invalid_candidate_moves": len(result["invalid_candidate_moves"]),
            }))
        except Exception as e:
            errors.append({"id": stem, "error": str(e)})
            print(json.dumps({
                "id": stem,
                "status": "error",
                "error": str(e),
            }))

    print(json.dumps({
        "puzzles_dir": str(puzzles_dir),
        "image_dir": str(image_dir),
        "model_dir": str(model_dir),
        "processed": processed,
        "errors": len(errors),
    }, indent=2))


if __name__ == "__main__":
    main()

# python batch_candidate_gen_plain.py --puzzles-dir puzzles --model-dir-name model_name --model google/gemma-4-31b-it --num-candidates 5