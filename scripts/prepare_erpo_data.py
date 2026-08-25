#!/usr/bin/env python3
"""Convert ERPO-5B0C's math JSON files to ROLL RLVR JSONL.

The conversion preserves every source row and applies the exact math prompt
suffix used by ``ERPO-5B0C/examples/format_prompt/math.jinja``.  Generated
files use one common schema so they can be loaded together by Hugging Face
Datasets and routed to ROLL's math-rule reward worker. AIME 2024 and 2025 also
get avg4 variants that repeat the complete 30-question set four times.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROMPT_SUFFIX = (
    " You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE put in \\boxed{}."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_row(item: dict[str, Any], dataset_name: str, index: int, split: str) -> dict[str, Any]:
    problem = str(item["problem"]).strip()
    prompt = problem + PROMPT_SUFFIX
    tag = "erpo_math_train" if split == "train" else f"erpo_{dataset_name}"
    stable_id = hashlib.sha256(
        json.dumps([dataset_name, index, problem, str(item["answer"])], ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    messages = [{"role": "user", "content": prompt}]
    return {
        "id": stable_id,
        "source": dataset_name,
        # Online prompt weighting is enabled for this reproduction. Keep a
        # neutral value for compatibility with the common RLVR schema.
        "difficulty": 1.0,
        "prompt": prompt,
        "messages": json.dumps(messages, ensure_ascii=False),
        "ground_truth": str(item["answer"]),
        "case_type": "",
        "test_case_function": "",
        "test_cases": "",
        "tag": tag,
        "original_level": str(item.get("level", item.get("difficulty", ""))),
    }


def convert_file(
    source: Path,
    destination: Path,
    split: str,
    *,
    dataset_name: str | None = None,
    repeats: int = 1,
) -> dict[str, Any]:
    with source.open("r", encoding="utf-8") as input_file:
        items = json.load(input_file)
    if not isinstance(items, list):
        raise TypeError(f"Expected a JSON array in {source}, got {type(items).__name__}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")

    dataset_name = dataset_name or source.stem
    repeated_items = items * repeats
    output_hash = hashlib.sha256()
    with destination.open("w", encoding="utf-8") as output_file:
        for index, item in enumerate(repeated_items):
            row = make_row(item, dataset_name=dataset_name, index=index, split=split)
            encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            output_file.write(encoded.decode("utf-8"))
            output_hash.update(encoded)

    return {
        "source": str(source),
        "destination": str(destination),
        "rows": len(repeated_items),
        "unique_source_rows": len(items),
        "repeats": repeats,
        "source_sha256": sha256_file(source),
        "destination_sha256": output_hash.hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("../ERPO-5B0C/data"),
        help="ERPO-5B0C data directory containing math_lvl3to5_8k.json and math_eval/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/erpo"),
        help="ROLL JSONL output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    sources = [
        (source_dir / "math_lvl3to5_8k.json", output_dir / "train/math_lvl3to5_8k.jsonl", "train", None, 1)
    ]
    sources.extend(
        (source, output_dir / f"eval/{source.stem}.jsonl", "eval", None, 1)
        for source in sorted((source_dir / "math_eval").glob("*.json"))
    )
    for year in (2024, 2025):
        dataset_name = f"aime_{year}_avg4"
        sources.append(
            (
                source_dir / f"math_eval/aime_{year}.json",
                output_dir / f"eval/{dataset_name}.jsonl",
                "eval",
                dataset_name,
                4,
            )
        )

    missing = [str(source) for source, _, _, _, _ in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing ERPO source data: {missing}")

    converted_files = []
    for source, destination, split, dataset_name, repeats in sources:
        converted = convert_file(
            source,
            destination,
            split,
            dataset_name=dataset_name,
            repeats=repeats,
        )
        converted["source"] = str(source.relative_to(source_dir))
        converted["destination"] = str(destination.relative_to(output_dir))
        converted_files.append(converted)

    manifest = {
        "source_root": "ERPO-5B0C/data",
        "prompt_suffix": PROMPT_SUFFIX,
        "files": converted_files,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        manifest_file.write("\n")

    for item in manifest["files"]:
        print(f"{item['rows']:>5}  {item['destination']}")


if __name__ == "__main__":
    main()
