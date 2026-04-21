#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List, Tuple

import chess


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


def process_json_file(json_path: Path) -> Dict[str, object]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fen = data.get("fen")
    if not fen:
        return {"id": json_path.stem, "status": "error", "error": "missing_fen"}

    move_data = data.get("move_data")
    if not isinstance(move_data, dict):
        return {"id": json_path.stem, "status": "error", "error": "missing_move_data"}

    candidate_moves = move_data.get("candidate_moves", [])
    if not isinstance(candidate_moves, list):
        return {"id": json_path.stem, "status": "error", "error": "candidate_moves_not_list"}

    board = chess.Board(fen)
    valid_moves, invalid_moves = validate_candidate_moves(board, candidate_moves)

    data["move_data"]["candidate_moves"] = dedupe_keep_order(candidate_moves)
    data["move_data"]["valid_candidate_moves"] = valid_moves
    data["move_data"]["invalid_candidate_moves"] = invalid_moves

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return {
        "id": data.get("id", json_path.stem),
        "status": "ok",
        "num_candidate_moves": len(data["move_data"]["candidate_moves"]),
        "num_valid_candidate_moves": len(valid_moves),
        "num_invalid_candidate_moves": len(invalid_moves),
    }


def main() -> None:
    folder = Path("puzzles/gemma-4-31b-it/text_only")

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {folder}")
        return

    processed = 0
    errors = 0

    for json_path in json_files:
        result = process_json_file(json_path)
        print(json.dumps(result))

        if result["status"] == "ok":
            processed += 1
        else:
            errors += 1

    print(json.dumps({
        "folder": str(folder),
        "processed": processed,
        "errors": errors,
        "total": len(json_files),
    }, indent=2))


if __name__ == "__main__":
    main()