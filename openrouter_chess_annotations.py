import argparse
import base64
import json
import mimetypes
import os
from typing import Any, Dict, List

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """
You are a chess vision annotation assistant.

You will be shown a chessboard image and told which side is to move.

Your job:
1. Infer the visible board state as carefully as possible.
2. Return structured JSON only.
3. First provide a piece map of the pieces you think are on the board.
4. Then provide useful board annotations for a renderer.

Rules:
- Return JSON only.
- No markdown.
- No commentary.
- No SAN.
- No FEN output.
- No duplicate arrows or duplicate square highlights.
- Use algebraic squares like e4, f7, a1.
- Use move-like arrow endpoints as from/to squares only.
- Only include annotations that are visually grounded in the current board.
- Keep the annotation set small and useful.

Arrow semantics:
- "candidate_move": a recommended move for the side to move
- "attack": a piece attacks a target square or piece
- "opponent_pressure": an opponent threat or pressure line
- "pin": a pin line

Square highlight semantics:
- "capture_target": a square containing an important target
- "hanging": a piece that looks loose or under-defended
- "checked_king": king currently under check
- "checker": piece currently giving check
- "pinned": pinned piece
- "pressure_target": square under notable opponent pressure
- "key_square": strategically important square

Keep output concise and machine-readable.
""".strip()


def image_to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def build_schema(
    max_pieces: int,
    max_arrows: int,
    max_highlights: int,
) -> Dict[str, Any]:
    square_pattern = "^[a-h][1-8]$"
    piece_symbol_pattern = "^[KQRBNPkqrbnp]$"
    arrow_type_pattern = "^(candidate_move|attack|opponent_pressure|pin)$"
    highlight_type_pattern = "^(capture_target|hanging|checked_king|checker|pinned|pressure_target|key_square)$"

    return {
        "name": "chess_annotations",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "side_to_move": {
                    "type": "string",
                    "enum": ["white", "black"]
                },
                "board_state": {
                    "type": "object",
                    "properties": {
                        "pieces": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "square": {
                                        "type": "string",
                                        "pattern": square_pattern
                                    },
                                    "piece": {
                                        "type": "string",
                                        "pattern": piece_symbol_pattern
                                    }
                                },
                                "required": ["square", "piece"],
                                "additionalProperties": False
                            },
                            "minItems": 1,
                            "maxItems": max_pieces
                        }
                    },
                    "required": ["pieces"],
                    "additionalProperties": False
                },
                "annotations": {
                    "type": "object",
                    "properties": {
                        "arrows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from": {
                                        "type": "string",
                                        "pattern": square_pattern
                                    },
                                    "to": {
                                        "type": "string",
                                        "pattern": square_pattern
                                    },
                                    "type": {
                                        "type": "string",
                                        "pattern": arrow_type_pattern
                                    }
                                },
                                "required": ["from", "to", "type"],
                                "additionalProperties": False
                            },
                            "maxItems": max_arrows
                        },
                        "highlights": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "square": {
                                        "type": "string",
                                        "pattern": square_pattern
                                    },
                                    "type": {
                                        "type": "string",
                                        "pattern": highlight_type_pattern
                                    }
                                },
                                "required": ["square", "type"],
                                "additionalProperties": False
                            },
                            "maxItems": max_highlights
                        }
                    },
                    "required": ["arrows", "highlights"],
                    "additionalProperties": False
                }
            },
            "required": ["side_to_move", "board_state", "annotations"],
            "additionalProperties": False
        }
    }


def build_user_prompt(
    side_to_move: str,
    max_arrows: int,
    max_highlights: int,
) -> str:
    return (
        f"The side to move is {side_to_move}. "
        f"Look at the chessboard image. "
        f"First infer the visible piece placement as a piece list. "
        f"Then return at most {max_arrows} arrows and at most {max_highlights} square highlights. "
        f"Choose annotations that would help a chess renderer visualize tactical ideas. "
        f"Return JSON only matching the schema."
    )


def parse_json_content(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content

    if isinstance(content, str):
        return json.loads(content)

    raise ValueError(f"Unexpected message content type: {type(content)}")


def request_chess_annotations(
    api_key: str,
    image_path: str,
    side_to_move: str,
    model: str = "google/gemma-4-31b-it:free",
    max_pieces: int = 32,
    max_arrows: int = 8,
    max_highlights: int = 8,
    reasoning_enabled: bool = True,
) -> Dict[str, Any]:
    image_data_url = image_to_data_url(image_path)
    schema = build_schema(
        max_pieces=max_pieces,
        max_arrows=max_arrows,
        max_highlights=max_highlights,
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_user_prompt(
                            side_to_move=side_to_move,
                            max_arrows=max_arrows,
                            max_highlights=max_highlights,
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        },
                    },
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": schema,
        },
        "plugins": [
            {"id": "response-healing"}
        ],
        "stream": False,
    }

    if reasoning_enabled:
        payload["reasoning"] = {"enabled": True}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Chess Annotation Generator",
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

    message = data["choices"][0]["message"]
    parsed = parse_json_content(message["content"])

    if "board_state" not in parsed or "annotations" not in parsed:
        raise ValueError("Missing required annotation fields in model response.")

    return {
        "model": model,
        "image_path": image_path,
        "side_to_move": side_to_move,
        "annotations_json": parsed,
        "raw_response": data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate parseable chess annotations from a board image using OpenRouter.")
    parser.add_argument("--image", required=True, help="Path to board image (png/jpg/webp/gif).")
    parser.add_argument("--side-to-move", required=True, choices=["white", "black"], help="Which side is to move.")
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"), help="OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.")
    parser.add_argument("--model", default="google/gemma-4-31b-it:free", help="Vision-capable OpenRouter model.")
    parser.add_argument("--max-arrows", type=int, default=8, help="Maximum arrows to request.")
    parser.add_argument("--max-highlights", type=int, default=8, help="Maximum highlighted squares to request.")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning mode.")
    parser.add_argument("--json-out", default="", help="Optional path to save parsed JSON output.")
    parser.add_argument("--raw-out", default="", help="Optional path to save raw API response JSON.")
    args = parser.parse_args()

    if not args.api_key:
        args.api_key = "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"

    result = request_chess_annotations(
        api_key=args.api_key,
        image_path=args.image,
        side_to_move=args.side_to_move,
        model=args.model,
        max_arrows=args.max_arrows,
        max_highlights=args.max_highlights,
        reasoning_enabled=not args.no_reasoning,
    )

    parsed_output = {
        "image_path": result["image_path"],
        "side_to_move": result["side_to_move"],
        "model": result["model"],
        "board_state": result["annotations_json"]["board_state"],
        "annotations": result["annotations_json"]["annotations"],
    }

    print(json.dumps(parsed_output, indent=2))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(parsed_output, f, indent=2)

    if args.raw_out:
        with open(args.raw_out, "w", encoding="utf-8") as f:
            json.dump(result["raw_response"], f, indent=2)


if __name__ == "__main__":
    main()