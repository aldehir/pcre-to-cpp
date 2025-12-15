#!/usr/bin/env python3
"""
Benchmark runner for PCRE-to-C++ vs STL regex vs PCRE2 comparison.

Uses HuggingFace datasets for benchmark inputs. Supports any text dataset
with configurable field extraction.

Orchestrates:
1. Loading text data from HuggingFace datasets
2. Generating C++ code from PCRE patterns
3. Building the C++ benchmark harness
4. Running benchmarks comparing generated C++ vs STL regex vs PCRE2
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


# Cache for loaded datasets to avoid re-downloading
_dataset_cache: dict[str, list[str]] = {}


def load_hf_dataset(input_config: dict, input_name: str) -> list[str]:
    """Load and chunk a HuggingFace dataset.

    Args:
        input_config: Configuration dict with dataset, subset, split, field, max_chars, chunk_size
        input_name: Name of this input (for caching and logging)

    Returns:
        List of text chunks for benchmarking
    """
    # Check cache first
    if input_name in _dataset_cache:
        print(f"Using cached dataset: {input_name}")
        return _dataset_cache[input_name]

    dataset_name = input_config.get("dataset")
    if not dataset_name:
        print(f"Error: Input '{input_name}' missing 'dataset' field", file=sys.stderr)
        return []

    subset = input_config.get("subset")
    split = input_config.get("split", "train")
    field = input_config.get("field", "text")
    max_chars = input_config.get("max_chars")
    chunk_size = input_config.get("chunk_size", 10000)

    # Build dataset identifier for logging
    dataset_id = f"{dataset_name}"
    if subset:
        dataset_id += f"/{subset}"
    dataset_id += f" ({split} split)"

    print(f"Loading HuggingFace dataset: {dataset_id}...")

    try:
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset '{dataset_id}': {e}", file=sys.stderr)
        return []

    print(f"Concatenating {len(dataset)} entries from field '{field}'...")

    # Concatenate text from dataset
    if max_chars:
        texts = []
        total_chars = 0
        for item in dataset:
            text = item.get(field, "")
            if not text:
                continue
            if total_chars + len(text) > max_chars:
                remaining = max_chars - total_chars
                texts.append(text[:remaining])
                break
            texts.append(text)
            total_chars += len(text)
        full_text = "\n".join(texts)
    else:
        full_text = "\n".join(item.get(field, "") for item in dataset)

    print(f"Total text length: {len(full_text):,} characters")

    # Split into chunks
    test_strings = []
    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i:i + chunk_size]
        if chunk.strip():
            test_strings.append(chunk)

    print(f"Split into {len(test_strings)} chunks of ~{chunk_size} characters each")

    # Cache the result
    _dataset_cache[input_name] = test_strings
    return test_strings


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PCRE-to-C++ vs STL regex vs PCRE2 using HuggingFace datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config file format (YAML):

inputs:
  wikitext-large:
    dataset: wikitext           # HuggingFace dataset name
    subset: wikitext-103-v1     # Dataset subset (optional)
    split: train                # Dataset split (default: train)
    field: text                 # Text field name (default: text)
    max_chars: 1000000          # Max chars to load (optional)
    chunk_size: 10000           # Chunk size (default: 10000)

benchmarks:
  - name: gpt2
    pcre_pattern: "..."
    stl_pattern: "..."
    inputs:
      - wikitext-large

Example usage:
    python benchmark.py                      # Run all benchmarks
    python benchmark.py -c my_benchmarks.yaml  # Run from specific file
    python benchmark.py -n gpt2              # Run only the 'gpt2' benchmark
    python benchmark.py -v                   # Verbose output
    python benchmark.py -o results/          # Save JSON results to directory
"""
    )
    parser.add_argument("--config", "-c", default="benchmarks.yaml",
                        help="YAML config file (default: benchmarks.yaml)")
    parser.add_argument("--name", "-n", help="Run only the benchmark with this name")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output (show per-test timing details)")
    parser.add_argument("--list", "-l", action="store_true", help="List available benchmarks")
    parser.add_argument("--output", "-o", help="Save JSON results to file or directory")

    args = parser.parse_args()

    # Load config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Create a benchmarks.yaml file or specify one with --config")
        sys.exit(1)

    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        print(f"Error: Expected dict in config file with 'inputs' and 'benchmarks' keys", file=sys.stderr)
        sys.exit(1)

    inputs_config = config.get("inputs", {})
    benchmarks = config.get("benchmarks", [])

    if not isinstance(benchmarks, list):
        print(f"Error: 'benchmarks' should be a list", file=sys.stderr)
        sys.exit(1)

    if args.list:
        print(f"Available benchmarks in {config_path}:")
        for b in benchmarks:
            name = b.get("name", "unnamed")
            pcre = b.get("pcre_pattern", "")[:40]
            stl = b.get("stl_pattern", "")[:40]
            input_names = ", ".join(b.get("inputs", []))
            print(f"  - {name}:")
            print(f"      PCRE: {pcre}...")
            print(f"      STL:  {stl}...")
            print(f"      inputs: [{input_names}]")
        print(f"\nAvailable inputs:")
        for input_name, input_cfg in inputs_config.items():
            dataset = input_cfg.get("dataset", "?")
            subset = input_cfg.get("subset", "")
            max_chars = input_cfg.get("max_chars")
            chars_str = f"{max_chars:,}" if max_chars else "all"
            print(f"  - {input_name}: {dataset}/{subset} ({chars_str} chars)")
        sys.exit(0)

    # Filter by name if specified
    if args.name:
        benchmarks = [b for b in benchmarks if b.get("name") == args.name]
        if not benchmarks:
            print(f"Error: No benchmark named '{args.name}' found")
            sys.exit(1)

    # Run benchmarks
    all_success = True
    total_runs = 0

    for benchmark in benchmarks:
        name = benchmark.get("name", "unnamed")
        pcre_pattern = benchmark.get("pcre_pattern")
        stl_pattern = benchmark.get("stl_pattern")
        input_names = benchmark.get("inputs", [])

        if not pcre_pattern:
            print(f"Error: Benchmark '{name}' missing 'pcre_pattern'", file=sys.stderr)
            all_success = False
            continue

        if not stl_pattern:
            print(f"Error: Benchmark '{name}' missing 'stl_pattern'", file=sys.stderr)
            all_success = False
            continue

        if not input_names:
            print(f"Warning: Benchmark '{name}' has no inputs specified", file=sys.stderr)
            continue

        # Run benchmark against each input separately
        for input_name in input_names:
            if input_name not in inputs_config:
                print(f"Error: Input '{input_name}' not found in inputs config", file=sys.stderr)
                all_success = False
                continue

            input_config = inputs_config[input_name]
            test_strings = load_hf_dataset(input_config, input_name)

            if not test_strings:
                print(f"Warning: No test strings loaded for input '{input_name}'", file=sys.stderr)
                continue

            # Build name includes input for separate results
            build_name = f"{name}_{input_name}"

            # Determine output path
            output_path = args.output if args.output else None

            if not run_benchmark_test(pcre_pattern, stl_pattern, test_strings,
                                      build_name, args.verbose, output_path):
                all_success = False
            total_runs += 1

    print(f"\n{'='*60}")
    if all_success:
        print(f"ALL {total_runs} BENCHMARK RUN(S) COMPLETED SUCCESSFULLY")
    else:
        print(f"SOME BENCHMARKS FAILED")
    print(f"{'='*60}")

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
