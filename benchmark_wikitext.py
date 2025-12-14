#!/usr/bin/env python3
"""
Benchmark runner using the WikiText dataset from HuggingFace.

Loads the wikitext dataset, concatenates all text, and benchmarks
PCRE-to-C++ generated code vs STL regex.
"""

import argparse
import sys
import yaml
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not installed. Install with: pip install datasets")
    sys.exit(1)

from benchmark_common import run_benchmark_test


def load_wikitext(split: str = "train", subset: str = "wikitext-103-v1", max_chars: int = None) -> str:
    """Load wikitext dataset and concatenate all text.

    Args:
        split: Dataset split to use (train, validation, test)
        subset: Which wikitext subset (wikitext-2-v1, wikitext-103-v1, etc.)
        max_chars: Maximum characters to load (None for all)

    Returns:
        Concatenated text from the dataset
    """
    print(f"Loading wikitext dataset: {subset} ({split} split)...")
    dataset = load_dataset("wikitext", subset, split=split, trust_remote_code=True)

    print(f"Concatenating {len(dataset)} entries...")

    if max_chars:
        # Build text up to max_chars
        texts = []
        total_chars = 0
        for item in dataset:
            text = item["text"]
            if total_chars + len(text) > max_chars:
                # Add partial text to reach max_chars
                remaining = max_chars - total_chars
                texts.append(text[:remaining])
                break
            texts.append(text)
            total_chars += len(text)
        full_text = "\n".join(texts)
    else:
        full_text = "\n".join(item["text"] for item in dataset)

    print(f"Total text length: {len(full_text):,} characters")
    return full_text


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PCRE-to-C++ vs STL regex vs PCRE2 using WikiText dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python benchmark_wikitext.py                          # Run all benchmarks with wikitext-103
    python benchmark_wikitext.py --subset wikitext-2-v1   # Use smaller wikitext-2 dataset
    python benchmark_wikitext.py -n gpt2                  # Run only the 'gpt2' benchmark
    python benchmark_wikitext.py --max-chars 1000000      # Limit to 1M characters
    python benchmark_wikitext.py -v                       # Verbose output
    python benchmark_wikitext.py -o results/              # Save JSON results to directory
"""
    )
    parser.add_argument("--config", "-c", default="benchmarks.yaml",
                        help="YAML config file with benchmark patterns (default: benchmarks.yaml)")
    parser.add_argument("--name", "-n", help="Run only the benchmark with this name")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output (show per-test timing details)")
    parser.add_argument("--list", "-l", action="store_true", help="List available benchmarks")
    parser.add_argument("--output", "-o", help="Save JSON results to file or directory")
    parser.add_argument("--subset", default="wikitext-103-v1",
                        choices=["wikitext-2-raw-v1", "wikitext-2-v1",
                                 "wikitext-103-raw-v1", "wikitext-103-v1"],
                        help="WikiText dataset subset (default: wikitext-103-v1)")
    parser.add_argument("--split", default="train",
                        choices=["train", "validation", "test"],
                        help="Dataset split to use (default: train)")
    parser.add_argument("--max-chars", type=int, default=None,
                        help="Maximum characters to load from dataset (default: all)")
    parser.add_argument("--chunk-size", type=int, default=10000,
                        help="Size of text chunks to benchmark (default: 10000)")

    args = parser.parse_args()

    # Load config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Create a benchmarks.yaml file or specify one with --config")
        sys.exit(1)

    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not isinstance(config, list):
        print(f"Error: Expected list in config file, got {type(config)}", file=sys.stderr)
        sys.exit(1)

    benchmarks = config

    if args.list:
        print(f"Available benchmarks in {config_path}:")
        for b in benchmarks:
            name = b.get("name", "unnamed")
            pcre = b.get("pcre_pattern", "")[:30]
            stl = b.get("stl_pattern", "")[:30]
            print(f"  - {name}:")
            print(f"      PCRE: {pcre}...")
            print(f"      STL:  {stl}...")
        sys.exit(0)

    # Filter by name if specified
    if args.name:
        benchmarks = [b for b in benchmarks if b.get("name") == args.name]
        if not benchmarks:
            print(f"Error: No benchmark named '{args.name}' found")
            sys.exit(1)

    # Load wikitext dataset
    full_text = load_wikitext(args.split, args.subset, args.max_chars)

    # Split into chunks for benchmarking
    chunk_size = args.chunk_size
    test_strings = []
    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i:i + chunk_size]
        if chunk.strip():  # Skip empty chunks
            test_strings.append(chunk)

    print(f"Split into {len(test_strings)} chunks of ~{chunk_size} characters each")

    # Run benchmarks
    all_success = True

    for benchmark in benchmarks:
        name = benchmark.get("name", "unnamed")
        pcre_pattern = benchmark.get("pcre_pattern")
        stl_pattern = benchmark.get("stl_pattern")

        if not pcre_pattern:
            print(f"Error: Benchmark '{name}' missing 'pcre_pattern'", file=sys.stderr)
            all_success = False
            continue

        if not stl_pattern:
            print(f"Error: Benchmark '{name}' missing 'stl_pattern'", file=sys.stderr)
            all_success = False
            continue

        # Use wikitext suffix for build name to avoid conflicts
        build_name = f"{name}_wikitext"

        # Determine output path for this benchmark
        output_path = None
        if args.output:
            output_path = args.output

        if not run_benchmark_test(pcre_pattern, stl_pattern, test_strings,
                                  build_name, args.verbose, output_path):
            all_success = False

    print(f"\n{'='*60}")
    if all_success:
        print(f"ALL {len(benchmarks)} BENCHMARK(S) COMPLETED SUCCESSFULLY")
    else:
        print(f"SOME BENCHMARKS FAILED")
    print(f"{'='*60}")

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
