#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import chess
import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

ANNOTATION_SYSTEM_PROMPT = """
You are a chess text-only annotation assistant.

You will be given a legal chess FEN string that fully specifies the board position and side to move.

Return JSON only with:
1. candidate_move_arrows: plausible candidate move arrows for the side to move
2. threat_arrows: opponent threats toward one of my pieces or an important threatened square
3. key_squares: important squares

Rules:
- Return JSON only
- No markdown
- No commentary
- No SAN
- Do not repeat the FEN
- Use algebraic squares like e4, f7, a1
- Use standard piece symbols: K Q R B N P for White, k q r b n p for Black
- No duplicate arrows
- No duplicate key squares
- Candidate move arrows should be plausible moves for the side to move
- Threat arrows should be plausible opponent threats
- If unsure, omit rather than guess
""".strip()

MOVE_FOLLOWUP_PROMPT = """
Using the same chess position and your earlier analysis, now return ONLY the best candidate moves.

Return JSON only in exactly this form:
{
  "candidate_moves": ["e2e4", "g1f3"]
}

Rules:
- Use UCI move notation only
- No markdown
- No commentary
- No extra keys besides candidate_moves
- No duplicate moves
- Return up to the requested number of moves
- Keep the moves consistent with the same FEN and your previous analysis in this conversation
""".strip()


def parse_json_content(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        return json.loads(content)
    raise ValueError(f"Unexpected message content type: {type(content)}")


def sanitize_model_name(model: str) -> str:
    name = model.strip().split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    return name or "model"


def output_is_complete(output_json_path: Path) -> bool:
    if not output_json_path.exists():
        return False
    try:
        with open(output_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        required = ["id", "fen", "side_to_move", "model", "annotation_data", "move_data"]
        if not all(k in data for k in required):
            return False
        if "candidate_moves" not in data["move_data"]:
            return False
        return True
    except Exception:
        return False


def call_openrouter(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Chess Text-Only Annotation + Move Baseline",
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


def build_annotation_schema(max_candidate_arrows: int, max_threat_arrows: int, max_key_squares: int) -> Dict[str, Any]:
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
        "name": "chess_text_only_annotation",
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


def build_move_schema(max_candidate_moves: int) -> Dict[str, Any]:
    uci_pattern = "^[a-h][1-8][a-h][1-8][qrbn]?$"
    return {
        "name": "chess_text_only_moves",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "candidate_moves": {
                    "type": "array",
                    "items": {"type": "string", "pattern": uci_pattern},
                    "maxItems": max_candidate_moves,
                }
            },
            "required": ["candidate_moves"],
            "additionalProperties": False,
        },
    }


def build_annotation_user_prompt(
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
        f"Return JSON only matching the schema."
    )


def request_annotation_data(
    api_key: str,
    model: str,
    fen: str,
    side_to_move: str,
    max_candidate_arrows: int,
    max_threat_arrows: int,
    max_key_squares: int,
    reasoning_enabled: bool,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_annotation_user_prompt(
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
            "json_schema": build_annotation_schema(
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

    assistant_message = {"role": "assistant", "content": message.get("content")}
    if "reasoning_details" in message:
        assistant_message["reasoning_details"] = message["reasoning_details"]

    return {
        "parsed_json": parsed,
        "assistant_message": assistant_message,
        "initial_messages": payload["messages"],
    }


def request_candidate_moves_followup(
    api_key: str,
    model: str,
    previous_messages: List[Dict[str, Any]],
    assistant_message: Dict[str, Any],
    max_candidate_moves: int,
    reasoning_enabled: bool,
) -> Dict[str, Any]:
    messages = list(previous_messages)
    messages.append(assistant_message)
    messages.append(
        {
            "role": "user",
            "content": f"{MOVE_FOLLOWUP_PROMPT}\nReturn up to {max_candidate_moves} moves.",
        }
    )

    payload = {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": build_move_schema(max_candidate_moves=max_candidate_moves),
        },
        "plugins": [{"id": "response-healing"}],
        "stream": False,
    }

    if reasoning_enabled:
        payload["reasoning"] = {"enabled": True}

    data = call_openrouter(api_key, payload)
    message = data["choices"][0]["message"]
    parsed = parse_json_content(message["content"])
    return {"parsed_json": parsed}


def process_one(
    input_annotation_json_path: Path,
    output_json_path: Path,
    api_key: str,
    model: str,
    max_candidate_arrows: int,
    max_threat_arrows: int,
    max_key_squares: int,
    max_candidate_moves: int,
    reasoning_enabled: bool,
) -> Dict[str, Any]:
    with open(input_annotation_json_path, "r", encoding="utf-8") as f:
        input_meta = json.load(f)

    fen = input_meta.get("fen")
    if not fen:
        raise ValueError(f"Missing top-level 'fen' in {input_annotation_json_path}")

    board = chess.Board(fen)
    side_to_move = "white" if board.turn == chess.WHITE else "black"

    annotation_result = request_annotation_data(
        api_key=api_key,
        model=model,
        fen=fen,
        side_to_move=side_to_move,
        max_candidate_arrows=max_candidate_arrows,
        max_threat_arrows=max_threat_arrows,
        max_key_squares=max_key_squares,
        reasoning_enabled=reasoning_enabled,
    )

    move_result = request_candidate_moves_followup(
        api_key=api_key,
        model=model,
        previous_messages=annotation_result["initial_messages"],
        assistant_message=annotation_result["assistant_message"],
        max_candidate_moves=max_candidate_moves,
        reasoning_enabled=reasoning_enabled,
    )

    output = {
        "id": input_annotation_json_path.stem,
        "input_annotation_json_path": str(input_annotation_json_path),
        "fen": fen,
        "side_to_move": side_to_move,
        "model": model,
        "baseline_type": "text_only_same_conversation",
        "annotation_data": annotation_result["parsed_json"],
        "move_data": move_result["parsed_json"],
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch run text-only same-conversation annotation + move generation over puzzles/annotations_json."
    )
    parser.add_argument("--puzzles-dir", required=True, help="Path to the root puzzles directory.")
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"), help="OpenRouter API key.")
    parser.add_argument("--model", default="google/gemma-4-31b-it", help="OpenRouter model.")
    parser.add_argument("--max-candidate-arrows", type=int, default=2)
    parser.add_argument("--max-threat-arrows", type=int, default=3)
    parser.add_argument("--max-key-squares", type=int, default=3)
    parser.add_argument("--max-candidate-moves", type=int, default=5)
    parser.add_argument("--no-reasoning", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers. Recommended 4-16 for paid OpenRouter models.")
    args = parser.parse_args()

    if not args.api_key:
        args.api_key = "..."

    puzzles_dir = Path(args.puzzles_dir)
    input_annotations_dir = puzzles_dir / "annotations_json"
    if not input_annotations_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {input_annotations_dir}")

    model_dir = puzzles_dir / sanitize_model_name(args.model) / "text_only"
    model_dir.mkdir(parents=True, exist_ok=True)

    input_json_paths = sorted(input_annotations_dir.glob("*.json"))
    if args.start_index:
        input_json_paths = input_json_paths[args.start_index:]
    if args.limit > 0:
        input_json_paths = input_json_paths[:args.limit]

    processed = 0
    skipped = 0
    errors = []

    work_items = []
    for input_annotation_json_path in input_json_paths:
        stem = input_annotation_json_path.stem
        output_json_path = model_dir / f"{stem}.json"

        if (not args.force) and output_is_complete(output_json_path):
            skipped += 1
            print(json.dumps({"id": stem, "status": "skipped_complete", "json": str(output_json_path)}))
            continue

        work_items.append((input_annotation_json_path, output_json_path))

    max_workers = max(1, min(args.workers, 16))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(
                process_one,
                input_annotation_json_path=input_annotation_json_path,
                output_json_path=output_json_path,
                api_key=args.api_key,
                model=args.model,
                max_candidate_arrows=args.max_candidate_arrows,
                max_threat_arrows=args.max_threat_arrows,
                max_key_squares=args.max_key_squares,
                max_candidate_moves=args.max_candidate_moves,
                reasoning_enabled=not args.no_reasoning,
            ): (input_annotation_json_path, output_json_path)
            for input_annotation_json_path, output_json_path in work_items
        }

        for future in as_completed(future_to_item):
            input_annotation_json_path, output_json_path = future_to_item[future]
            stem = input_annotation_json_path.stem
            try:
                result = future.result()
                processed += 1
                print(json.dumps({
                    "id": stem,
                    "status": "ok",
                    "json": str(output_json_path),
                    "num_candidate_moves": len(result["move_data"].get("candidate_moves", [])),
                }))
            except Exception as e:
                errors.append({"id": stem, "error": str(e)})
                print(json.dumps({"id": stem, "status": "error", "error": str(e)}))

    summary = {
        "puzzles_dir": str(puzzles_dir),
        "model_output_dir": str(model_dir),
        "processed": processed,
        "skipped": skipped,
        "errors": len(errors),
        "workers": max(1, min(args.workers, 16)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
