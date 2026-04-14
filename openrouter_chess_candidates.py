import argparse
import base64
import json
import mimetypes
import os
from typing import Any, Dict, List

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """
You are a chess vision assistant.
Given a chessboard image and the side to move, return a short list of strong candidate moves.

Rules:
- Return UCI moves only.
- Return JSON only.
- No markdown.
- No commentary.
- No SAN.
- No duplicate moves.
- Only include moves for the given side to move.
- Prefer legal-looking, high-quality candidate moves.
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
                    "description": "Candidate chess moves in UCI notation only.",
                    "items": {
                        "type": "string",
                        "pattern": "^[a-h][1-8][a-h][1-8][qrbn]?$"
                    },
                    "minItems": 1,
                    "maxItems": max_candidates
                }
            },
            "required": ["candidate_moves"],
            "additionalProperties": False
        }
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


def request_candidate_moves(
    api_key: str,
    image_path: str,
    side_to_move: str,
    model: str = "google/gemma-4-31b-it:free",
    max_candidates: int = 5,
    reasoning_enabled: bool = True,
) -> Dict[str, Any]:
    image_data_url = image_to_data_url(image_path)
    schema = build_schema(max_candidates)

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
                        "text": build_user_prompt(side_to_move, max_candidates),
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
        # Optional OpenRouter headers
        "HTTP-Referer": "https://localhost",
        "X-Title": "Chess Candidate Move Generator",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate parseable chess candidate moves from a board image using OpenRouter.")
    parser.add_argument("--image", required=True, help="Path to board image (png/jpg/webp/gif).")
    parser.add_argument("--side-to-move", required=True, choices=["white", "black"], help="Which side is to move.")
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"), help="OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.")
    parser.add_argument("--model", default="google/gemma-4-31b-it:free", help="Vision-capable OpenRouter model.")
    parser.add_argument("--num-candidates", type=int, default=5, help="How many candidate moves to request.")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning mode.")
    parser.add_argument("--json-out", default="", help="Optional path to save parsed JSON output.")
    parser.add_argument("--raw-out", default="", help="Optional path to save raw API response JSON.")
    args = parser.parse_args()

    if not args.api_key:
        args.api_key = "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"

    result = request_candidate_moves(
        api_key=args.api_key,
        image_path=args.image,
        side_to_move=args.side_to_move,
        model=args.model,
        max_candidates=args.num_candidates,
        reasoning_enabled=not args.no_reasoning,
    )

    parsed_output = {
        "image_path": result["image_path"],
        "side_to_move": result["side_to_move"],
        "model": result["model"],
        "candidate_moves": result["candidate_moves"],
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