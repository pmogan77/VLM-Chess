#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_random_arrows(data: dict) -> dict:
    candidate = len(data.get("candidate_move_arrows", []))
    threat = len(data.get("threat_arrows", []))

    # fallback if needed
    if candidate == 0 and threat == 0:
        arrows = data.get("annotation_bundle", {}).get("arrows", [])
        candidate = sum(1 for a in arrows if a.get("kind") == "candidate_move")
        threat = sum(1 for a in arrows if a.get("kind") == "threat")

    return {
        "candidate": candidate,
        "threat": threat,
        "total": candidate + threat,
    }


def count_full_arrows(data: dict) -> dict:
    arrows = data.get("arrows", [])

    candidate = sum(1 for a in arrows if a.get("kind") == "candidate_move")
    threat = sum(1 for a in arrows if a.get("kind") == "threat")

    return {
        "candidate": candidate,
        "threat": threat,
        "total": len(arrows),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare arrow counts between random annotation JSONs and full/all annotation JSONs."
    )

    parser.add_argument(
        "--random-dir",
        default="puzzles/random/annotations_json",
        help="Directory containing random annotation JSON files.",
    )

    parser.add_argument(
        "--full-dir",
        default="puzzles/annotations_json",
        help="Directory containing full/all annotation JSON files.",
    )

    parser.add_argument(
        "--out-csv",
        default="arrow_count_comparison.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()

    random_dir = Path(args.random_dir)
    full_dir = Path(args.full_dir)
    out_csv = Path(args.out_csv)

    if not random_dir.is_dir():
        raise FileNotFoundError(f"Missing random dir: {random_dir}")

    if not full_dir.is_dir():
        raise FileNotFoundError(f"Missing full dir: {full_dir}")

    random_files = {p.stem: p for p in random_dir.glob("*.json")}
    full_files = {p.stem: p for p in full_dir.glob("*.json")}

    common_ids = sorted(set(random_files) & set(full_files))
    only_random = sorted(set(random_files) - set(full_files))
    only_full = sorted(set(full_files) - set(random_files))

    rows = []

    for puzzle_id in common_ids:
        random_data = load_json(random_files[puzzle_id])
        full_data = load_json(full_files[puzzle_id])

        r = count_random_arrows(random_data)
        f = count_full_arrows(full_data)

        rows.append({
            "id": puzzle_id,

            "random_candidate_arrows": r["candidate"],
            "random_threat_arrows": r["threat"],
            "random_total_arrows": r["total"],

            "full_candidate_arrows": f["candidate"],
            "full_threat_arrows": f["threat"],
            "full_total_arrows": f["total"],

            "full_minus_random_total": f["total"] - r["total"],
            "full_more_than_random": f["total"] > r["total"],
            "same_arrow_count": f["total"] == r["total"],
            "random_more_than_full": r["total"] > f["total"],
        })

    fieldnames = [
        "id",
        "random_candidate_arrows",
        "random_threat_arrows",
        "random_total_arrows",
        "full_candidate_arrows",
        "full_threat_arrows",
        "full_total_arrows",
        "full_minus_random_total",
        "full_more_than_random",
        "same_arrow_count",
        "random_more_than_full",
    ]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)

    if n == 0:
        print("No matching puzzle IDs found.")
        print(f"Only random: {len(only_random)}")
        print(f"Only full: {len(only_full)}")
        return

    full_more = sum(1 for row in rows if row["full_more_than_random"])
    same = sum(1 for row in rows if row["same_arrow_count"])
    random_more = sum(1 for row in rows if row["random_more_than_full"])

    print()
    print("=== Arrow Count Comparison Summary ===")
    print(f"Compared puzzles: {n}")
    print(f"Full more than random: {full_more} / {n} = {full_more / n:.2%}")
    print(f"Same arrow count: {same} / {n} = {same / n:.2%}")
    print(f"Random more than full: {random_more} / {n} = {random_more / n:.2%}")

    print()
    print("=== Averages ===")
    print(f"Avg random total arrows: {mean(row['random_total_arrows'] for row in rows):.2f}")
    print(f"Avg full total arrows: {mean(row['full_total_arrows'] for row in rows):.2f}")
    print(f"Avg full minus random: {mean(row['full_minus_random_total'] for row in rows):.2f}")

    print()
    print("=== Missing Files ===")
    print(f"Only in random dir: {len(only_random)}")
    print(f"Only in full dir: {len(only_full)}")

    print()
    print(f"Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()

# python compare_arrow_counts.py --random-dir puzzles/random/annotations_json --full-dir puzzles/annotations_json --out-csv arrow_count_comparison.csv