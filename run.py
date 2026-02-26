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


def _run_cmd(cmd: list[str], verbose: bool = False, error_label: str = "Command failed",
             **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, capturing output unless verbose."""
    if verbose:
        result = subprocess.run(cmd, text=True, **kwargs)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0 and not verbose:
        stderr = getattr(result, 'stderr', '') or ''
        stdout = getattr(result, 'stdout', '') or ''
        detail = (stderr + '\n' + stdout).strip()
        if detail:
            print(f"{error_label}: {detail}", file=sys.stderr)
    return result


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

    result = _run_cmd(cmd, verbose, error_label="Error generating C++ code")
    if result.returncode != 0:
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
    result = _run_cmd(configure_cmd, verbose, error_label="CMake configure failed", cwd=build_dir)
    if result.returncode != 0:
        return False

    # Build
    build_cmd = ["cmake", "--build", ".", "--config", "Release"]
    result = _run_cmd(build_cmd, verbose, error_label="Build failed", cwd=build_dir)
    if result.returncode != 0:
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
            encoding='utf-8',
            errors='replace'
        )
    else:
        result = subprocess.run(
            [str(exe_path)],
            input=input_json,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        # Print stderr on error when not verbose
        if result.stderr and result.returncode != 0:
            print(result.stderr, file=sys.stderr)

    # In test mode, non-zero exit means test failures (but output is still valid)
    if mode != "test" and result.returncode != 0:
        stderr = getattr(result, 'stderr', '') or ''
        print(f"Harness execution failed{': ' + stderr.strip() if stderr.strip() else ''}", file=sys.stderr)
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"Failed to parse harness output: {e}", file=sys.stderr)
        return None


def print_summary(results: dict, name: str, mode: str):
    """Print a formatted summary of results."""
    summary = results.get("summary", {})
    label = "TEST" if mode == "test" else "BENCHMARK"

    print(f"\n{'='*60}")
    print(f"{label} SUMMARY: {name}")
    print(f"{'='*60}")

    if mode == "test":
        all_passed = summary.get('all_passed', False)
        mismatches = summary.get('token_mismatches', 0)
        if all_passed:
            print("All tests PASSED")
        else:
            print(f"FAILED: {mismatches} test(s) had token mismatches vs PCRE2")
    else:
        print(f"Total Generated C++ time:  {summary.get('total_generated_ms', 0):.3f}ms")
        print(f"Total STL Regex time:      {summary.get('total_stl_ms', 0):.3f}ms")
        if summary.get('total_boost_ms', 0) > 0:
            print(f"Total Boost Regex time:    {summary.get('total_boost_ms', 0):.3f}ms")
        print(f"Total PCRE2 time:          {summary.get('total_pcre2_ms', 0):.3f}ms")
        print(f"Speedup vs STL:            {summary.get('average_speedup_vs_stl', 0):.1f}x")
        if summary.get('average_speedup_vs_boost', 0) > 0:
            print(f"Speedup vs Boost:          {summary.get('average_speedup_vs_boost', 0):.1f}x")
        print(f"Speedup vs PCRE2:          {summary.get('average_speedup_vs_pcre2', 0):.1f}x")
        print(f"STL Regex failures:        {summary.get('stl_failures', 0)}")
        if summary.get('boost_failures', 0) > 0:
            print(f"Boost Regex failures:      {summary.get('boost_failures', 0)}")
        print(f"PCRE2 failures:            {summary.get('pcre2_failures', 0)}")
        print(f"Token mismatches vs PCRE2: {summary.get('token_mismatches', 0)}")


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


_MODE_DEFAULTS = {
    "test": {"iterations": 1, "success_msg": "TEST(S) PASSED", "fail_msg": "SOME TESTS FAILED"},
    "bench": {"iterations": 50, "success_msg": "BENCHMARK(S) COMPLETED SUCCESSFULLY", "fail_msg": "SOME BENCHMARKS FAILED"},
}


def _plot_timing_bars(bench_data: list[tuple[str, dict]], output_dir: Path,
                      fmt: str, dpi: int):
    """Horizontal grouped bar chart showing total time per engine per pattern."""
    import matplotlib.pyplot as plt
    import numpy as np

    engine_keys = [
        ("Generated", "total_generated_ms", "tab:blue"),
        ("PCRE2 (JIT)", "total_pcre2_ms", "tab:red"),
        ("Boost", "total_boost_ms", "tab:green"),
        ("STL", "total_stl_ms", "tab:orange"),
    ]
    # Only include engines that have non-zero time in at least one file
    engines = [(label, key, color) for label, key, color in engine_keys
               if any(d["summary"].get(key, 0) > 0 for _, d in bench_data)]

    if not engines:
        print("Warning: No timing data found, skipping timing chart")
        return

    # Strip _benchmark suffix from labels
    labels = [name.removesuffix("_benchmark") for name, _ in bench_data]
    y = np.arange(len(labels))
    height = 0.8 / len(engines)

    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 1.5)))

    for i, (engine_label, key, color) in enumerate(engines):
        values = [d["summary"].get(key, 0) for _, d in bench_data]
        offset = (i - len(engines) / 2 + 0.5) * height
        bars = ax.barh(y + offset, values, height, label=engine_label, color=color)
        # Value labels at end of bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_width() + ax.get_xlim()[1] * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.0f}ms", ha="left", va="center", fontsize=8)

    ax.set_xlabel("Total Time (ms)")
    ax.set_title("Total Execution Time by Engine (lower is better)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.legend(loc="lower right")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.15)  # room for labels
    fig.tight_layout()

    out_path = output_dir / f"timing_bars.{fmt}"
    fig.savefig(out_path, dpi=dpi)
    print(f"Saved: {out_path}")


def _plot_throughput_vs_size(bench_data: list[tuple[str, dict]], output_dir: Path,
                             fmt: str, dpi: int):
    """Throughput vs input size: median line with IQR band, bucketed by size."""
    import matplotlib.pyplot as plt
    import numpy as np

    engine_configs = [
        ("Generated", "generated", "tab:blue"),
        ("PCRE2 (JIT)", "pcre2", "tab:red"),
        ("Boost", "boost_regex", "tab:green"),
        ("STL", "stl_regex", "tab:orange"),
    ]

    # Collect all (size, throughput) points per engine across all files
    series: dict[str, list[tuple[float, float]]] = {label: [] for label, *_ in engine_configs}

    for _, data in bench_data:
        iterations = data.get("iterations", 1)
        for r in data.get("results", []):
            size_cp = r.get("input_length_codepoints")
            if size_cp is None:
                size_cp = len(r.get("input", ""))
            if size_cp == 0:
                continue

            for label, key, _ in engine_configs:
                engine = r.get(key, {})
                if not engine.get("success", False):
                    continue
                time_ms = engine.get("time_ms", 0)
                if time_ms <= 0:
                    continue
                throughput = (size_cp * iterations) / time_ms
                series[label].append((size_cp, throughput))

    has_data = {label for label, pts in series.items() if pts}
    if not has_data:
        print("Warning: No throughput data found, skipping throughput chart")
        return

    # Create log-spaced bins for bucketing
    all_sizes = []
    for pts in series.values():
        all_sizes.extend(s for s, _ in pts)
    if not all_sizes:
        return
    bin_edges = np.logspace(np.log10(max(1, min(all_sizes))),
                            np.log10(max(all_sizes)), 20)

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, key, color in engine_configs:
        pts = series[label]
        if not pts:
            continue
        sizes = np.array([s for s, _ in pts])
        thrpts = np.array([t for _, t in pts])

        # Bucket into bins, compute median + IQR
        bin_indices = np.digitize(sizes, bin_edges)
        bin_centers, medians, q25s, q75s = [], [], [], []
        for bi in range(1, len(bin_edges) + 1):
            mask = bin_indices == bi
            if mask.sum() < 3:
                continue
            bucket = thrpts[mask]
            center = np.median(sizes[mask])
            bin_centers.append(center)
            medians.append(np.median(bucket))
            q25s.append(np.percentile(bucket, 25))
            q75s.append(np.percentile(bucket, 75))

        if not bin_centers:
            continue
        bc = np.array(bin_centers)
        med = np.array(medians)
        ax.plot(bc, med, color=color, label=label, linewidth=2)
        ax.fill_between(bc, q25s, q75s, alpha=0.15, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Input Size (codepoints)")
    ax.set_ylabel("Throughput (codepoints / ms)")
    ax.set_title("Throughput vs Input Size (median with IQR band)")
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / f"throughput_vs_size.{fmt}"
    fig.savefig(out_path, dpi=dpi)
    print(f"Saved: {out_path}")


def cmd_plot(args) -> int:
    """Generate charts from benchmark JSON result files."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend by default
    except ImportError:
        print("Error: matplotlib is required for plotting. Install with: pip install matplotlib",
              file=sys.stderr)
        return 1

    # Load and filter bench results
    bench_data: list[tuple[str, dict]] = []
    for filepath in args.files:
        p = Path(filepath)
        if not p.exists():
            print(f"Warning: File not found, skipping: {p}", file=sys.stderr)
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not read {p}: {e}", file=sys.stderr)
            continue

        if data.get("mode") != "bench":
            print(f"Skipping non-bench file: {p.name} (mode={data.get('mode', '?')})")
            continue

        bench_data.append((p.stem, data))

    if not bench_data:
        print("Error: No benchmark JSON files found", file=sys.stderr)
        return 1

    print(f"Loaded {len(bench_data)} benchmark file(s): {', '.join(n for n, _ in bench_data)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_timing_bars(bench_data, output_dir, args.format, args.dpi)
    _plot_throughput_vs_size(bench_data, output_dir, args.format, args.dpi)

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()

    return 0


def cmd_run(args, mode: str):
    """Run tests or benchmarks for all configured patterns."""
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

    mode_cfg = _MODE_DEFAULTS[mode]
    all_success = True
    total_runs = 0
    iterations = args.iterations or mode_cfg["iterations"]

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

        for input_name in input_names:
            if input_name not in inputs_config:
                print(f"Error: Input '{input_name}' not found in inputs config", file=sys.stderr)
                all_success = False
                continue

            test_strings = load_hf_dataset(inputs_config[input_name], input_name)

            if not test_strings:
                print(f"Warning: No test strings loaded for input '{input_name}'", file=sys.stderr)
                continue

            if not run_single_pattern(pcre_pattern, stl_pattern, test_strings,
                                      name, mode, iterations, args.verbose, args.output, args.rebuild):
                all_success = False
            total_runs += 1

    print(f"\n{'='*60}")
    if all_success:
        print(f"ALL {total_runs} {mode_cfg['success_msg']}")
    else:
        print(mode_cfg["fail_msg"])
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
  plot    Generate charts from benchmark JSON results

Examples:
  python run.py test                    # Run all tests
  python run.py test -n gpt2            # Test specific pattern
  python run.py bench                   # Run all benchmarks
  python run.py bench --iterations 100  # More iterations
  python run.py bench -o results/       # Save JSON results
  python run.py bench --rebuild         # Force full rebuild (no incremental)
  python run.py plot results/*.json -d results/charts/  # Generate charts
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
    test_parser.set_defaults(func=lambda args: cmd_run(args, "test"))

    # Bench subcommand
    bench_parser = subparsers.add_parser("bench", help="Run performance benchmarks")
    add_common_args(bench_parser)
    bench_parser.set_defaults(func=lambda args: cmd_run(args, "bench"))

    # Plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Generate charts from benchmark JSON results")
    plot_parser.add_argument("files", nargs="+", help="JSON result files from bench runs")
    plot_parser.add_argument("--output-dir", "-d", default=".", help="Directory to save charts (default: .)")
    plot_parser.add_argument("--format", "-f", default="png", choices=["png", "svg", "pdf"],
                             help="Output image format (default: png)")
    plot_parser.add_argument("--dpi", type=int, default=150, help="Image DPI (default: 150)")
    plot_parser.add_argument("--no-show", action="store_true", help="Skip plt.show() (headless)")
    plot_parser.set_defaults(func=cmd_plot)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
