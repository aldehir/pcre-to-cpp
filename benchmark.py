#!/usr/bin/env python3
"""
Benchmark runner for PCRE-to-C++ vs STL regex comparison.

Orchestrates:
1. Generating C++ code from PCRE patterns
2. Building the C++ benchmark harness
3. Running benchmarks comparing generated C++ vs STL regex approximations
"""

import argparse
import io
import json
import os
import subprocess
import sys
import yaml
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Paths
ROOT_DIR = Path(__file__).parent
HARNESS_DIR = ROOT_DIR / "test-harness"


def get_build_dir(name: str) -> Path:
    """Get build directory for a specific benchmark."""
    return HARNESS_DIR / "builds" / "benchmarks" / name


def get_generated_file(name: str) -> Path:
    """Get generated C++ file path for a specific benchmark."""
    return HARNESS_DIR / "generated" / "benchmarks" / f"{name}.cpp"


def generate_cpp(pattern: str, name: str) -> bool:
    """Generate C++ code from PCRE pattern."""
    output_file = get_generated_file(name)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "pcre_to_cpp.py"),
            "--pattern", pattern,
            "--name", "test",
            "--output", str(output_file)
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error generating C++ code: {result.stderr}", file=sys.stderr)
        return False

    print(f"Generated: {output_file}")
    return True


def build_cpp(name: str) -> bool:
    """Build the C++ benchmark harness."""
    build_dir = get_build_dir(name)
    build_dir.mkdir(parents=True, exist_ok=True)

    # Clean CMake cache if pattern changed (force reconfigure)
    cache_file = build_dir / "CMakeCache.txt"
    if cache_file.exists():
        cache_file.unlink()

    # Pattern file path relative to HARNESS_DIR (where CMakeLists.txt is)
    pattern_file = f"generated/benchmarks/{name}.cpp"

    # Configure (build dir is 3 levels deep: builds/benchmarks/{name}/)
    result = subprocess.run(
        ["cmake", f"-DPATTERN_FILE={pattern_file}", "../../.."],
        cwd=build_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"CMake configure failed: {result.stderr}", file=sys.stderr)
        return False

    # Build
    result = subprocess.run(
        ["cmake", "--build", "."],
        cwd=build_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}\n{result.stdout}", file=sys.stderr)
        return False

    print("Build successful")
    return True


def find_benchmark_exe(name: str) -> Path | None:
    """Find the benchmark executable."""
    build_dir = get_build_dir(name)
    candidates = [
        build_dir / "benchmark.exe",
        build_dir / "benchmark",
        build_dir / "Debug" / "benchmark.exe",
        build_dir / "Release" / "benchmark.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def run_benchmark(name: str, stl_pattern: str, test_strings: list[str], verbose: bool = False) -> dict | None:
    """Run benchmark comparing generated C++ vs STL regex."""
    exe_path = find_benchmark_exe(name)
    if not exe_path:
        print(f"Could not find benchmark executable in {get_build_dir(name)}", file=sys.stderr)
        return None

    input_json = json.dumps({
        "pretokenizer": stl_pattern,
        "strings": test_strings
    })

    result = subprocess.run(
        [str(exe_path)],
        input=input_json,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )

    # Print stderr (human-readable output) if verbose or on error
    if result.stderr:
        if verbose or result.returncode != 0:
            print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"Benchmark execution failed", file=sys.stderr)
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"Failed to parse benchmark output: {e}", file=sys.stderr)
        return None


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


def print_summary(results: dict, name: str):
    """Print a formatted summary of benchmark results."""
    summary = results.get("summary", {})

    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY: {name}")
    print(f"{'='*60}")
    print(f"Total Generated C++ time: {summary.get('total_generated_ms', 0):.3f}ms")
    print(f"Total STL Regex time:     {summary.get('total_stl_ms', 0):.3f}ms")

    stl_failures = summary.get('stl_failures', 0)
    if stl_failures > 0:
        print(f"STL Regex failures:       {stl_failures}")

    avg_speedup = summary.get('average_speedup', 0)
    if avg_speedup > 0:
        print(f"Average speedup:          {avg_speedup:.1f}x")

    # Check for token mismatches
    mismatches = 0
    for r in results.get("results", []):
        gen_tokens = r.get("generated", {}).get("tokens", [])
        stl_tokens = r.get("stl_regex", {}).get("tokens", [])
        if gen_tokens != stl_tokens and r.get("stl_regex", {}).get("success", False):
            mismatches += 1

    if mismatches > 0:
        print(f"Token mismatches:         {mismatches} (STL pattern may differ from PCRE)")


def run_benchmark_test(pcre_pattern: str, stl_pattern: str, test_strings: list[str],
                       name: str, verbose: bool = False, output_json: str = None) -> bool:
    """Run a complete benchmark cycle."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"PCRE pattern:  {pcre_pattern[:50]}{'...' if len(pcre_pattern) > 50 else ''}")
    print(f"STL pattern:   {stl_pattern[:50]}{'...' if len(stl_pattern) > 50 else ''}")
    print(f"Test strings:  {len(test_strings)}")
    print(f"{'='*60}")

    # Step 1: Generate C++ from PCRE pattern
    print("\n[1/3] Generating C++ code from PCRE pattern...")
    if not generate_cpp(pcre_pattern, name):
        return False

    # Step 2: Build
    print("\n[2/3] Building benchmark executable...")
    if not build_cpp(name):
        return False

    # Step 3: Run benchmark with STL pattern
    print("\n[3/3] Running benchmark...")
    results = run_benchmark(name, stl_pattern, test_strings, verbose)
    if results is None:
        return False

    # Print summary
    print_summary(results, name)

    # Save JSON output if requested
    if output_json:
        output_path = Path(output_json)
        # If output_json is a directory, create a file named after the test
        if output_path.is_dir() or output_json.endswith('/') or output_json.endswith('\\'):
            output_path = Path(output_json) / f"{name}_benchmark.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PCRE-to-C++ vs STL regex",
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
    all_results = []

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
