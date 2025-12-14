#!/usr/bin/env python3
"""
Benchmark runner for PCRE-to-C++ vs STL regex comparison.

Orchestrates:
1. Generating C++ code from PCRE patterns
2. Building the C++ benchmark harness
3. Running benchmarks comparing generated C++ vs STL regex approximations
"""

import argparse
import sys
import yaml
from pathlib import Path

from benchmark_common import HARNESS_DIR, run_benchmark_test


def load_test_inputs(input_names: list[str]) -> list[str]:
    """Load and combine test strings from input files."""
    all_tests = []
    for name in input_names:
        input_file = HARNESS_DIR / "inputs" / f"{name}.yaml"
        if not input_file.exists():
            print(f"Warning: Input file not found: {input_file}", file=sys.stderr)
            continue
        with open(input_file, encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if isinstance(data, list):
                all_tests.extend(data)
            else:
                print(f"Warning: Expected list in {input_file}, got {type(data)}", file=sys.stderr)
    return all_tests


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PCRE-to-C++ vs STL regex vs PCRE2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config file format (YAML):
- name: test_name
  pcre_pattern: "\\\\p{L}+"           # Pattern for C++ code generation
  stl_pattern: "[A-Za-z]+"           # STL regex approximation
  inputs:
    - natural_language
    - code

Input files are loaded from test-harness/inputs/ (top-level YAML lists).

Example usage:
    python run_benchmark.py                      # Run all benchmarks from benchmarks.yaml
    python run_benchmark.py -c my_benchmarks.yaml  # Run from specific file
    python run_benchmark.py -n gpt2              # Run only the 'gpt2' benchmark
    python run_benchmark.py -v                   # Verbose output (show per-test details)
    python run_benchmark.py -o results/          # Save JSON results to directory
"""
    )
    parser.add_argument("--config", "-c", default="benchmarks.yaml",
                        help="YAML config file with benchmark patterns (default: benchmarks.yaml)")
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
            inputs = ", ".join(b.get("inputs", []))
            print(f"  - {name}:")
            print(f"      PCRE: {pcre}...")
            print(f"      STL:  {stl}...")
            print(f"      inputs: [{inputs}]")
        sys.exit(0)

    # Filter by name if specified
    if args.name:
        benchmarks = [b for b in benchmarks if b.get("name") == args.name]
        if not benchmarks:
            print(f"Error: No benchmark named '{args.name}' found")
            sys.exit(1)

    # Run benchmarks
    all_success = True

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

        test_strings = load_test_inputs(input_names)
        if not test_strings:
            print(f"Warning: No test strings loaded for {name}", file=sys.stderr)
            continue

        # Determine output path for this benchmark
        output_path = None
        if args.output:
            output_path = args.output

        if not run_benchmark_test(pcre_pattern, stl_pattern, test_strings,
                                  name, args.verbose, output_path):
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
