from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path("puzzles/gemma-4-31b-it")

# Folder name -> relative path from ROOT
FOLDER_MAP = {
    "gold_moves": "gold_moves",
    "model_moves": "model_moves",
    "plain_moves": "plain_moves",
    "random_moves": "random_moves",
    "text_moves": "text_only/moves",
}

EXPECTED_COUNT = 99


def safe_load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return None, f"read_error: {e}"

    if text == "":
        return None, "empty_file"

    try:
        data = json.loads(text)
    except Exception as e:
        return None, f"invalid_json: {e}"

    if not isinstance(data, dict):
        return None, "top_level_not_object"

    return data, None


def list_json_stems(folder: Path) -> List[str]:
    return sorted(p.stem for p in folder.glob("*.json"))


def audit_one_file(path: Path) -> Dict[str, Any] | None:
    data, load_error = safe_load_json(path)
    if load_error is not None:
        return {
            "file": path.name,
            "issue_type": load_error,
        }

    issues: List[str] = []

    required_top = [
        "id",
        "fen",
        "side_to_move",
        "model",
        "baseline_type",
        "move_data",
    ]
    for key in required_top:
        if key not in data:
            issues.append(f"missing_top_level_key:{key}")

    move_data = data.get("move_data")
    if not isinstance(move_data, dict):
        issues.append("move_data_not_object")
        return {
            "file": path.name,
            "issue_type": "schema_issue",
            "details": issues,
        }

    if "candidate_moves" not in move_data:
        issues.append("missing_move_data_key:candidate_moves")
        return {
            "file": path.name,
            "issue_type": "schema_issue",
            "details": issues,
        }

    candidate_moves = move_data.get("candidate_moves")
    if not isinstance(candidate_moves, list):
        issues.append("candidate_moves_not_list")
        return {
            "file": path.name,
            "issue_type": "schema_issue",
            "details": issues,
        }

    if len(candidate_moves) == 0:
        issues.append("candidate_moves_empty")

    # Optional validation fields
    valid_moves = move_data.get("valid_candidate_moves")
    invalid_moves = move_data.get("invalid_candidate_moves")

    if valid_moves is None:
        issues.append("missing_move_data_key:valid_candidate_moves")
    elif not isinstance(valid_moves, list):
        issues.append("valid_candidate_moves_not_list")

    if invalid_moves is None:
        issues.append("missing_move_data_key:invalid_candidate_moves")
    elif not isinstance(invalid_moves, list):
        issues.append("invalid_candidate_moves_not_list")

    # Null-ish move entries
    if isinstance(candidate_moves, list):
        for i, mv in enumerate(candidate_moves):
            if mv is None:
                issues.append(f"candidate_move_null_at_index:{i}")
            elif not isinstance(mv, str):
                issues.append(f"candidate_move_not_string_at_index:{i}")

    # Mild consistency checks if present
    if isinstance(valid_moves, list):
        for i, mv in enumerate(valid_moves):
            if mv is None:
                issues.append(f"valid_candidate_move_null_at_index:{i}")
            elif not isinstance(mv, str):
                issues.append(f"valid_candidate_move_not_string_at_index:{i}")

    if isinstance(invalid_moves, list):
        for i, mv in enumerate(invalid_moves):
            if mv is None:
                issues.append(f"invalid_candidate_move_null_at_index:{i}")

    if issues:
        return {
            "file": path.name,
            "issue_type": "schema_issue",
            "details": issues,
        }

    return None


def audit_folder(folder_name: str, rel_path: str) -> Dict[str, Any]:
    folder = ROOT / rel_path

    result: Dict[str, Any] = {
        "folder_name": folder_name,
        "folder_path": str(folder),
        "exists": folder.is_dir(),
    }

    if not folder.is_dir():
        result["summary"] = "missing_folder"
        return result

    json_files = sorted(folder.glob("*.json"))
    stems = [p.stem for p in json_files]

    result["num_json_files"] = len(json_files)
    result["has_expected_count"] = len(json_files) == EXPECTED_COUNT

    # Expected ids are inferred from annotations_json, which is the cleanest source of truth.
    annotations_dir = Path("puzzles/annotations_json")
    if annotations_dir.is_dir():
        expected_ids = sorted(p.stem for p in annotations_dir.glob("*.json"))
    else:
        expected_ids = []

    result["expected_count_from_annotations"] = len(expected_ids)

    if expected_ids:
        expected_set = set(expected_ids)
        actual_set = set(stems)
        result["missing_ids"] = sorted(expected_set - actual_set)
        result["extra_ids"] = sorted(actual_set - expected_set)
    else:
        result["missing_ids"] = []
        result["extra_ids"] = []

    file_issues: List[Dict[str, Any]] = []
    for path in json_files:
        issue = audit_one_file(path)
        if issue is not None:
            file_issues.append(issue)

    result["num_files_with_issues"] = len(file_issues)
    result["file_issues"] = file_issues

    empty_move_files = [
        issue["file"]
        for issue in file_issues
        if issue.get("issue_type") == "schema_issue"
        and "details" in issue
        and "candidate_moves_empty" in issue["details"]
    ]
    result["files_with_empty_candidate_moves"] = empty_move_files

    if len(json_files) == EXPECTED_COUNT and not result["missing_ids"] and not file_issues:
        result["summary"] = "ok"
    else:
        result["summary"] = "needs_attention"

    return result


def main() -> None:
    all_results = []
    for folder_name, rel_path in FOLDER_MAP.items():
        all_results.append(audit_folder(folder_name, rel_path))

    payload = {
        "root": str(ROOT),
        "expected_count": EXPECTED_COUNT,
        "folders": all_results,
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

# looks inside folder puzzles/gemma-4-31b-it
# folders = gold_moves, model_moves, plain_moves, random_moves, text_only/moves
# for each folder, checks:
# - folder exists
# - number of json files
# - expected count of json files (99)
# - expected ids from annotations_json
# - for each json file:
#   - can it be read and parsed as json?
#   - does it have the required top-level keys?


# python audit/audit.py