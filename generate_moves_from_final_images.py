#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chess
import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MOVE_SYSTEM_PROMPT = """
You are a chess move prediction assistant.

You will be shown an ALREADY-ANNOTATED chessboard image.
The annotations on the image are meaningful and must guide your move prediction.

Interpret the visual annotations as follows:

- Green arrows represent candidate moves for the side to move.
  These are the main move ideas already selected by an earlier analysis stage.
  Treat them as strong move priors.

- Orange arrows represent opponent threats, pressure, or tactical dangers.
  Use them to understand what the side to move may need to respond to, exploit, or avoid.

- Blue highlighted squares represent key squares.
  These may be tactically important, strategically important, contested, attacking, defensive, or strong destination squares.

Your task:
- Predict the best next moves for the side to move
- Use the annotations as guidance
- Keep your moves consistent with the arrows and highlighted squares when possible
- Prefer moves that follow a green candidate arrow unless there is a strong chess reason not to
- Use orange threat arrows to understand defensive needs or tactical opportunities
- Use blue key squares to understand important targets, outposts, king attack zones, or defensive squares

Important:
- The image is not a raw board only. It is an analysis board with prior guidance already drawn on it.
- Do not ignore the arrows and highlights.
- Do not invent unrelated moves unless the annotations are clearly misleading.
- If multiple green arrows are present, prioritize the strongest move among them.
- If a green arrow appears legal and positionally sensible, it should be strongly preferred.
- If a move responds to an orange threat while also matching a green arrow or blue square, that is especially strong.

Output rules:
- Return JSON only
- No markdown
- No commentary
- No SAN
- No FEN
- Use UCI move format only
- No duplicate moves
- Return only the requested key
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


def image_to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def build_move_schema(max_candidate_moves: int) -> Dict[str, Any]:
    uci_pattern = "^[a-h][1-8][a-h][1-8][qrbn]?$"
    return {
        "name": "chess_moves_from_annotated_image",
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

def build_move_user_prompt(side_to_move: str, max_candidate_moves: int) -> str:
    return (
        f"The side to move is {side_to_move}. "
        f"This image already contains analysis annotations. "
        f"Green arrows show candidate moves for the side to move. "
        f"Orange arrows show opponent threats or pressure. "
        f"Blue highlighted squares show important squares. "
        f"Predict up to {max_candidate_moves} best next moves in UCI format, "
        f"using the annotations as strong guidance. "
        f"Prefer moves that are consistent with the drawn candidate arrows unless there is a strong chess reason not to. "
        f"Return JSON only."
    )

def call_openrouter(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Chess Final-Image Move Prediction",
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


def request_candidate_moves_from_annotated_image(
    api_key: str,
    image_path: str,
    side_to_move: str,
    model: str,
    max_candidate_moves: int,
    reasoning_enabled: bool,
) -> Dict[str, Any]:
    image_data_url = image_to_data_url(image_path)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": MOVE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_move_user_prompt(
                            side_to_move=side_to_move,
                            max_candidate_moves=max_candidate_moves,
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
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
    return {"parsed_json": parsed, "raw_response": data}


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def validate_candidate_moves(board: chess.Board, items: List[str]) -> Tuple[List[str], List[Dict[str, str]]]:
    valid: List[str] = []
    invalid: List[Dict[str, str]] = []

    for move_str in dedupe_keep_order(items):
        try:
            move = chess.Move.from_uci(move_str)
        except ValueError:
            invalid.append({"move": move_str, "reason": "invalid_uci"})
            continue

        if move in board.legal_moves:
            valid.append(move_str)
        else:
            invalid.append({"move": move_str, "reason": "illegal_move"})

    return valid, invalid


def output_is_complete(output_json_path: Path) -> bool:
    if not output_json_path.exists():
        return False
    try:
        with open(output_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        required = [
            "id",
            "image_path",
            "input_annotation_json_path",
            "side_to_move",
            "fen",
            "model",
            "baseline_type",
            "move_data",
        ]
        if not all(k in data for k in required):
            return False
        if "candidate_moves" not in data["move_data"]:
            return False
        return True
    except Exception:
        return False


def resolve_input_paths(puzzles_dir: Path, model: str, image_subdir: str) -> Tuple[Path, Path, Path]:
    model_dir = puzzles_dir / sanitize_model_name(model)
    image_dir = model_dir / image_subdir
    input_annotations_dir = puzzles_dir / "annotations_json"
    output_dir = model_dir / "moves_from_final_images"
    return image_dir, input_annotations_dir, output_dir


def process_one(
    image_path: Path,
    input_annotation_json_path: Path,
    output_json_path: Path,
    api_key: str,
    model: str,
    max_candidate_moves: int,
    reasoning_enabled: bool,
    validate_moves: bool,
) -> Dict[str, Any]:
    with open(input_annotation_json_path, "r", encoding="utf-8") as f:
        input_meta = json.load(f)

    fen = input_meta.get("fen")
    if not fen:
        raise ValueError(f"Missing top-level 'fen' in {input_annotation_json_path}")

    board = chess.Board(fen)
    side_to_move = "white" if board.turn == chess.WHITE else "black"

    move_result = request_candidate_moves_from_annotated_image(
        api_key=api_key,
        image_path=str(image_path),
        side_to_move=side_to_move,
        model=model,
        max_candidate_moves=max_candidate_moves,
        reasoning_enabled=reasoning_enabled,
    )

    candidate_moves = dedupe_keep_order(move_result["parsed_json"].get("candidate_moves", []))

    output = {
        "id": image_path.stem,
        "image_path": str(image_path),
        "input_annotation_json_path": str(input_annotation_json_path),
        "side_to_move": side_to_move,
        "fen": fen,
        "model": model,
        "baseline_type": "image_final_to_moves",
        "move_data": {
            "candidate_moves": candidate_moves,
        },
    }

    if validate_moves:
        valid_moves, invalid_moves = validate_candidate_moves(board, candidate_moves)
        output["move_data"]["valid_candidate_moves"] = valid_moves
        output["move_data"]["invalid_candidate_moves"] = invalid_moves

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch run move prediction from existing final annotated board images."
    )
    parser.add_argument("--puzzles-dir", required=True, help="Path to the root puzzles directory.")
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"), help="OpenRouter API key.")
    parser.add_argument("--model", default="google/gemma-4-31b-it", help="Vision-capable OpenRouter model.")
    parser.add_argument(
        "--image-subdir",
        default="annotated_boards_final",
        help="Model subdirectory containing the generated images, relative to puzzles/<model>/ .",
    )
    parser.add_argument("--max-candidate-moves", type=int, default=5)
    parser.add_argument("--no-reasoning", action="store_true")
    parser.add_argument("--no-validate", action="store_true", help="Skip legal move validation against FEN.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers. Good range is usually 4 to 16.")
    args = parser.parse_args()

    if not args.api_key:
        from dotenv import load_dotenv
        load_dotenv()
        args.api_key = os.getenv("OPENROUTER_API_KEY")

    puzzles_dir = Path(args.puzzles_dir)
    image_dir, input_annotations_dir, output_dir = resolve_input_paths(
        puzzles_dir=puzzles_dir,
        model=args.model,
        image_subdir=args.image_subdir,
    )

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not input_annotations_dir.is_dir():
        raise FileNotFoundError(f"Missing input JSON directory: {input_annotations_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.png"))
    if args.start_index:
        image_paths = image_paths[args.start_index:]
    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    processed = 0
    skipped = 0
    errors: List[Dict[str, str]] = []
    work_items: List[Tuple[Path, Path, Path]] = []

    for image_path in image_paths:
        stem = image_path.stem
        input_annotation_json_path = input_annotations_dir / f"{stem}.json"
        output_json_path = output_dir / f"{stem}.json"

        if not input_annotation_json_path.exists():
            errors.append({"id": stem, "error": f"Missing input JSON: {input_annotation_json_path}"})
            print(json.dumps({"id": stem, "status": "error", "error": "missing_input_json"}))
            continue

        if (not args.force) and output_is_complete(output_json_path):
            skipped += 1
            print(json.dumps({"id": stem, "status": "skipped_complete", "json": str(output_json_path)}))
            continue

        work_items.append((image_path, input_annotation_json_path, output_json_path))

    max_workers = max(1, min(args.workers, 16))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(
                process_one,
                image_path=image_path,
                input_annotation_json_path=input_annotation_json_path,
                output_json_path=output_json_path,
                api_key=args.api_key,
                model=args.model,
                max_candidate_moves=args.max_candidate_moves,
                reasoning_enabled=not args.no_reasoning,
                validate_moves=not args.no_validate,
            ): (image_path, output_json_path)
            for image_path, input_annotation_json_path, output_json_path in work_items
        }

        for future in as_completed(future_to_item):
            image_path, output_json_path = future_to_item[future]
            stem = image_path.stem
            try:
                result = future.result()
                processed += 1
                move_data = result.get("move_data", {})
                print(json.dumps({
                    "id": stem,
                    "status": "ok",
                    "json": str(output_json_path),
                    "num_candidate_moves": len(move_data.get("candidate_moves", [])),
                    "num_valid_candidate_moves": len(move_data.get("valid_candidate_moves", move_data.get("candidate_moves", []))),
                }))
            except Exception as e:
                errors.append({"id": stem, "error": str(e)})
                print(json.dumps({"id": stem, "status": "error", "error": str(e)}))

    summary = {
        "puzzles_dir": str(puzzles_dir),
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "processed": processed,
        "skipped": skipped,
        "errors": len(errors),
        "workers": max_workers,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


# python generate_moves_from_final_images.py --puzzles-dir puzzles --model google/gemma-4-31b-it --workers 8 --max-candidate-moves 5 --limit 1