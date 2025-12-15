#!/usr/bin/env python3
"""
Unified test and benchmark runner for PCRE-to-C++ converter.

Orchestrates:
1. Loading text data from HuggingFace datasets
2. Generating C++ code from PCRE patterns
3. Building the C++ harness
4. Running tests (correctness) or benchmarks (performance)

Usage:
    python run.py test                    # Run all tests
    python run.py test -n gpt2            # Test specific pattern
    python run.py bench                   # Run all benchmarks
    python run.py bench -n gpt2           # Benchmark specific pattern
    python run.py bench --iterations 100  # More iterations
"""

import argparse
import io
import json
import subprocess
import sys
import yaml
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not installed. Install with: pip install datasets")
    sys.exit(1)

# Paths
ROOT_DIR = Path(__file__).parent
HARNESS_DIR = ROOT_DIR / "test-harness"

# Cache for loaded datasets to avoid re-downloading
_dataset_cache: dict[str, list[str]] = {}


# =============================================================================
# Dataset Loading
# =============================================================================

def load_hf_dataset(input_config: dict, input_name: str) -> list[str]:
    """Load a HuggingFace dataset for testing/benchmarking.

    Two modes:
        (a) Individual samples: Use each dataset entry as a test case.
            Config: max_samples (optional) - limit number of samples
        (b) Concatenate and chunk: Combine entries, then split into chunks.
            Config: chunk_size (required), max_chars (optional) - limit total chars before chunking

    Args:
        input_config: Configuration dict with dataset, subset, split, field, and mode options
        input_name: Name of this input (for caching and logging)

    Returns:
        List of text strings for testing/benchmarking
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
    # Mode (a): use individual samples
    max_samples = input_config.get("max_samples")
    # Mode (b): concatenate, limit, and chunk
    max_chars = input_config.get("max_chars")
    chunk_size = input_config.get("chunk_size")

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

    if chunk_size:
        # Mode (b): Concatenate, limit by max_chars, chunk by chunk_size
        print(f"Concatenating entries from field '{field}'...")
        texts = []
        total_chars = 0
        for item in dataset:
            text = item.get(field, "")
            if not text:
                continue
            if max_chars and total_chars + len(text) > max_chars:
                texts.append(text[:max_chars - total_chars])
                break
            texts.append(text)
            total_chars += len(text)
        full_text = "\n".join(texts)
        print(f"Total: {len(full_text):,} characters")

        test_strings = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
        test_strings = [s for s in test_strings if s.strip()]
        print(f"Split into {len(test_strings)} chunks of ~{chunk_size} chars")
    else:
        # Mode (a): Use individual samples, limit by max_samples
        limit = max_samples or len(dataset)
        test_strings = []
        for item in dataset:
            if len(test_strings) >= limit:
                break
            text = item.get(field, "")
            if text and text.strip():
                test_strings.append(text)
        print(f"Using {len(test_strings)} samples from field '{field}'")

    # Cache the result
    _dataset_cache[input_name] = test_strings
    return test_strings


# =============================================================================
# Build Orchestration
# =============================================================================

def get_build_dir(name: str) -> Path:
    """Get build directory for a specific pattern."""
    return HARNESS_DIR / "builds" / name


def get_generated_file(name: str) -> Path:
    """Get generated C++ file path for a specific pattern."""
    return HARNESS_DIR / "generated" / f"{name}.cpp"


def generate_cpp(pattern: str, name: str, verbose: bool = False) -> bool:
    """Generate C++ code from PCRE pattern."""
    output_file = get_generated_file(name)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT_DIR / "pcre_to_cpp.py"),
        "--pattern", pattern,
        "--name", "test",
        "--output", str(output_file)
    ]

    if verbose:
        result = subprocess.run(cmd, text=True)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if not verbose and result.stderr:
            print(f"Error generating C++ code: {result.stderr}", file=sys.stderr)
        return False

    print(f"Generated: {output_file}")
    return True


def build_cpp(name: str, verbose: bool = False, rebuild: bool = False) -> bool:
    """Build the C++ harness."""
    build_dir = get_build_dir(name)
    build_dir.mkdir(parents=True, exist_ok=True)

    # Clean CMake cache if rebuild requested (force reconfigure)
    if rebuild:
        cache_file = build_dir / "CMakeCache.txt"
        if cache_file.exists():
            cache_file.unlink()
            print("Removed CMakeCache.txt (forcing rebuild)")

    # Pattern file path relative to HARNESS_DIR (where CMakeLists.txt is)
    pattern_file = f"generated/{name}.cpp"

    # Configure (build dir is 2 levels deep: builds/{name}/)
    configure_cmd = ["cmake", f"-DPATTERN_FILE={pattern_file}", "-DCMAKE_BUILD_TYPE=Release", "../.."]
    if verbose:
        result = subprocess.run(configure_cmd, cwd=build_dir, text=True)
    else:
        result = subprocess.run(configure_cmd, cwd=build_dir, capture_output=True, text=True)

    if result.returncode != 0:
        if not verbose and result.stderr:
            print(f"CMake configure failed: {result.stderr}", file=sys.stderr)
        return False

    # Build
    build_cmd = ["cmake", "--build", ".", "--config", "Release"]
    if verbose:
        result = subprocess.run(build_cmd, cwd=build_dir, text=True)
    else:
        result = subprocess.run(build_cmd, cwd=build_dir, capture_output=True, text=True)

    if result.returncode != 0:
        if not verbose and (result.stderr or result.stdout):
            print(f"Build failed: {result.stderr}\n{result.stdout}", file=sys.stderr)
        return False

    print("Build successful")
    return True


def find_runner_exe(name: str) -> Path | None:
    """Find the runner executable."""
    build_dir = get_build_dir(name)
    candidates = [
        build_dir / "runner.exe",
        build_dir / "runner",
        build_dir / "Debug" / "runner.exe",
        build_dir / "Release" / "runner.exe",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


# =============================================================================
# Test/Benchmark Execution
# =============================================================================

def run_harness(name: str, mode: str, stl_pattern: str, pcre_pattern: str,
                test_strings: list[str], iterations: int, verbose: bool = False) -> dict | None:
    """Run the C++ harness in test or bench mode."""
    exe_path = find_runner_exe(name)
    if not exe_path:
        print(f"Could not find runner executable in {get_build_dir(name)}", file=sys.stderr)
        return None

    input_json = json.dumps({
        "mode": mode,
        "iterations": iterations,
        "pretokenizer": stl_pattern,
        "pcre_pattern": pcre_pattern,
        "strings": test_strings
    })

    # Always capture stdout (JSON output), but stream stderr when verbose
    if verbose:
        result = subprocess.run(
            [str(exe_path)],
            input=input_json,
            stdout=subprocess.PIPE,
            stderr=None,  # Inherit from parent - streams in real-time
            text=True,
            encoding='utf-8'
        )
    else:
        result = subprocess.run(
            [str(exe_path)],
            input=input_json,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        # Print stderr on error when not verbose
        if result.stderr and result.returncode != 0:
            print(result.stderr, file=sys.stderr)

    # In test mode, non-zero exit means test failures (but output is still valid)
    if mode != "test" and result.returncode != 0:
        print(f"Harness execution failed", file=sys.stderr)
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"Failed to parse harness output: {e}", file=sys.stderr)
        return None


def print_summary(results: dict, name: str, mode: str):
    """Print a formatted summary of results."""
    summary = results.get("summary", {})

    print(f"\n{'='*60}")
    if mode == "test":
        print(f"TEST SUMMARY: {name}")
    else:
        print(f"BENCHMARK SUMMARY: {name}")
    print(f"{'='*60}")

    if mode == "test":
        all_passed = summary.get('all_passed', False)
        mismatches = summary.get('token_mismatches', 0)
        if all_passed:
            print("All tests PASSED")
        else:
            print(f"FAILED: {mismatches} test(s) had token mismatches vs PCRE2")
    else:
        print(f"Total Generated C++ time: {summary.get('total_generated_ms', 0):.3f}ms")
        print(f"Total STL Regex time:     {summary.get('total_stl_ms', 0):.3f}ms")
        print(f"Total PCRE2 time:         {summary.get('total_pcre2_ms', 0):.3f}ms")

        stl_failures = summary.get('stl_failures', 0)
        if stl_failures > 0:
            print(f"STL Regex failures:       {stl_failures}")

        pcre2_failures = summary.get('pcre2_failures', 0)
        if pcre2_failures > 0:
            print(f"PCRE2 failures:           {pcre2_failures}")

        avg_speedup_stl = summary.get('average_speedup_vs_stl', 0)
        if avg_speedup_stl > 0:
            print(f"Speedup vs STL:           {avg_speedup_stl:.1f}x")

        avg_speedup_pcre2 = summary.get('average_speedup_vs_pcre2', 0)
        if avg_speedup_pcre2 > 0:
            print(f"Speedup vs PCRE2:         {avg_speedup_pcre2:.1f}x")

        mismatches = summary.get('token_mismatches', 0)
        if mismatches > 0:
            print(f"Token mismatches vs PCRE2: {mismatches}")


def run_single_pattern(pcre_pattern: str, stl_pattern: str, test_strings: list[str],
                       name: str, mode: str, iterations: int, verbose: bool = False,
                       output_json: str = None, rebuild: bool = False) -> bool:
    """Run a complete test/benchmark cycle for a single pattern."""
    print(f"\n{'='*60}")
    print(f"{'Test' if mode == 'test' else 'Benchmark'}: {name}")
    print(f"Mode: {mode}, Iterations: {iterations}")
    print(f"PCRE pattern:  {pcre_pattern[:50]}{'...' if len(pcre_pattern) > 50 else ''}")
    print(f"STL pattern:   {stl_pattern[:50]}{'...' if len(stl_pattern) > 50 else ''}")
    print(f"Test strings:  {len(test_strings)}")
    print(f"{'='*60}")

    # Step 1: Generate C++ from PCRE pattern
    print("\n[1/3] Generating C++ code from PCRE pattern...")
    if not generate_cpp(pcre_pattern, name, verbose):
        return False

    # Step 2: Build
    print("\n[2/3] Building harness executable...")
    if not build_cpp(name, verbose, rebuild):
        return False

    # Step 3: Run harness
    print(f"\n[3/3] Running {'tests' if mode == 'test' else 'benchmarks'}...")
    results = run_harness(name, mode, stl_pattern, pcre_pattern, test_strings, iterations, verbose)
    if results is None:
        return False

    # Print summary
    print_summary(results, name, mode)

    # Save JSON output if requested
    if output_json:
        output_path = Path(output_json)
        # If output_json is a directory, create a file named after the pattern
        if output_path.is_dir() or output_json.endswith('/') or output_json.endswith('\\'):
            suffix = "test" if mode == "test" else "benchmark"
            output_path = Path(output_json) / f"{name}_{suffix}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Return success based on test results
    if mode == "test":
        return results.get("summary", {}).get("all_passed", False)
    return True


# =============================================================================
# CLI Commands
# =============================================================================

def load_config(config_path: Path) -> dict:
    """Load and validate configuration file."""
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Create a config.yaml file or specify one with --config")
        sys.exit(1)

    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        print(f"Error: Expected dict in config file with 'inputs' and 'benchmarks' keys", file=sys.stderr)
        sys.exit(1)

    return config


def list_patterns(config: dict, config_path: Path):
    """List available patterns from config."""
    inputs_config = config.get("inputs", {})
    benchmarks = config.get("benchmarks", [])

    print(f"Available patterns in {config_path}:")
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
        chunk_size = input_cfg.get("chunk_size")
        if chunk_size:
            max_chars = input_cfg.get("max_chars")
            chars_str = f"{max_chars:,}" if max_chars else "all"
            print(f"  - {input_name}: {dataset}/{subset} (chunk mode: {chars_str} chars, {chunk_size} chunk)")
        else:
            max_samples = input_cfg.get("max_samples")
            samples_str = f"{max_samples}" if max_samples else "all"
            print(f"  - {input_name}: {dataset}/{subset} (sample mode: {samples_str} samples)")


def cmd_test(args):
    """Run tests (correctness verification)."""
    config = load_config(Path(args.config))

    if args.list:
        list_patterns(config, Path(args.config))
        return 0

    inputs_config = config.get("inputs", {})
    benchmarks = config.get("benchmarks", [])

    if not isinstance(benchmarks, list):
        print(f"Error: 'benchmarks' should be a list", file=sys.stderr)
        return 1

    # Filter by name if specified
    if args.name:
        benchmarks = [b for b in benchmarks if b.get("name") == args.name]
        if not benchmarks:
            print(f"Error: No pattern named '{args.name}' found")
            return 1

    # Run tests
    all_success = True
    total_runs = 0
    iterations = args.iterations if args.iterations else 1  # Default 1 for tests

    for benchmark in benchmarks:
        name = benchmark.get("name", "unnamed")
        pcre_pattern = benchmark.get("pcre_pattern")
        stl_pattern = benchmark.get("stl_pattern", "")
        input_names = benchmark.get("inputs", [])

        if not pcre_pattern:
            print(f"Error: Pattern '{name}' missing 'pcre_pattern'", file=sys.stderr)
            all_success = False
            continue

        if not input_names:
            print(f"Warning: Pattern '{name}' has no inputs specified", file=sys.stderr)
            continue

        # Run test against each input separately
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
            build_name = name

            if not run_single_pattern(pcre_pattern, stl_pattern, test_strings,
                                      build_name, "test", iterations, args.verbose, args.output, args.rebuild):
                all_success = False
            total_runs += 1

    print(f"\n{'='*60}")
    if all_success:
        print(f"ALL {total_runs} TEST(S) PASSED")
    else:
        print(f"SOME TESTS FAILED")
    print(f"{'='*60}")

    return 0 if all_success else 1


def cmd_bench(args):
    """Run benchmarks (performance comparison)."""
    config = load_config(Path(args.config))

    if args.list:
        list_patterns(config, Path(args.config))
        return 0

    inputs_config = config.get("inputs", {})
    benchmarks = config.get("benchmarks", [])

    if not isinstance(benchmarks, list):
        print(f"Error: 'benchmarks' should be a list", file=sys.stderr)
        return 1

    # Filter by name if specified
    if args.name:
        benchmarks = [b for b in benchmarks if b.get("name") == args.name]
        if not benchmarks:
            print(f"Error: No pattern named '{args.name}' found")
            return 1

    # Run benchmarks
    all_success = True
    total_runs = 0
    iterations = args.iterations if args.iterations else 50  # Default 50 for benchmarks

    for benchmark in benchmarks:
        name = benchmark.get("name", "unnamed")
        pcre_pattern = benchmark.get("pcre_pattern")
        stl_pattern = benchmark.get("stl_pattern", "")
        input_names = benchmark.get("inputs", [])

        if not pcre_pattern:
            print(f"Error: Pattern '{name}' missing 'pcre_pattern'", file=sys.stderr)
            all_success = False
            continue

        if not input_names:
            print(f"Warning: Pattern '{name}' has no inputs specified", file=sys.stderr)
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
            build_name = name

            if not run_single_pattern(pcre_pattern, stl_pattern, test_strings,
                                      build_name, "bench", iterations, args.verbose, args.output, args.rebuild):
                all_success = False
            total_runs += 1

    print(f"\n{'='*60}")
    if all_success:
        print(f"ALL {total_runs} BENCHMARK(S) COMPLETED SUCCESSFULLY")
    else:
        print(f"SOME BENCHMARKS FAILED")
    print(f"{'='*60}")

    return 0 if all_success else 1


def main():
    parser = argparse.ArgumentParser(
        description="PCRE-to-C++ test and benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  test    Run correctness tests (generated vs PCRE2)
  bench   Run performance benchmarks (generated vs STL vs PCRE2)

Examples:
  python run.py test                    # Run all tests
  python run.py test -n gpt2            # Test specific pattern
  python run.py bench                   # Run all benchmarks
  python run.py bench --iterations 100  # More iterations
  python run.py bench -o results/       # Save JSON results
  python run.py bench --rebuild         # Force full rebuild (no incremental)
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments
    def add_common_args(p):
        p.add_argument("--config", "-c", default="config.yaml",
                       help="YAML config file (default: config.yaml)")
        p.add_argument("--name", "-n", help="Run only the pattern with this name")
        p.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
        p.add_argument("--list", "-l", action="store_true", help="List available patterns")
        p.add_argument("--output", "-o", help="Save JSON results to file or directory")
        p.add_argument("--iterations", "-i", type=int, help="Number of iterations (default: 1 for test, 50 for bench)")
        p.add_argument("--rebuild", "-r", action="store_true", help="Force rebuild by removing CMakeCache.txt")

    # Test subcommand
    test_parser = subparsers.add_parser("test", help="Run correctness tests")
    add_common_args(test_parser)
    test_parser.set_defaults(func=cmd_test)

    # Bench subcommand
    bench_parser = subparsers.add_parser("bench", help="Run performance benchmarks")
    add_common_args(bench_parser)
    bench_parser.set_defaults(func=cmd_bench)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
