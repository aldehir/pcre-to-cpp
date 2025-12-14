#!/usr/bin/env python3
"""
Common utilities for benchmark runners.

Shared code for generating C++, building, and running benchmarks.
"""

import io
import json
import subprocess
import sys
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


def build_cpp(name: str, verbose: bool = False) -> bool:
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
    configure_cmd = ["cmake", f"-DPATTERN_FILE={pattern_file}", "-DCMAKE_BUILD_TYPE=Release", "../../.."]
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


def run_benchmark(name: str, stl_pattern: str, pcre_pattern: str, test_strings: list[str], verbose: bool = False) -> dict | None:
    """Run benchmark comparing generated C++ vs STL regex vs PCRE2."""
    exe_path = find_benchmark_exe(name)
    if not exe_path:
        print(f"Could not find benchmark executable in {get_build_dir(name)}", file=sys.stderr)
        return None

    input_json = json.dumps({
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

    if result.returncode != 0:
        print(f"Benchmark execution failed", file=sys.stderr)
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"Failed to parse benchmark output: {e}", file=sys.stderr)
        return None


def print_summary(results: dict, name: str):
    """Print a formatted summary of benchmark results."""
    summary = results.get("summary", {})

    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY: {name}")
    print(f"{'='*60}")
    print(f"Total Generated C++ time: {summary.get('total_generated_ms', 0):.3f}ms")
    print(f"Total STL Regex time:     {summary.get('total_stl_ms', 0):.3f}ms")
    print(f"Total PCRE2 time:         {summary.get('total_pcre2_ms', 0):.3f}ms")

    stl_failures = summary.get('stl_failures', 0)
    if stl_failures > 0:
        print(f"STL Regex failures:       {stl_failures}")

    pcre2_failures = summary.get('pcre2_failures', 0)
    if pcre2_failures > 0:
        print(f"PCRE2 failures:           {pcre2_failures}")

    avg_speedup_stl = summary.get('average_speedup_vs_stl', summary.get('average_speedup', 0))
    if avg_speedup_stl > 0:
        print(f"Speedup vs STL:           {avg_speedup_stl:.1f}x")

    avg_speedup_pcre2 = summary.get('average_speedup_vs_pcre2', 0)
    if avg_speedup_pcre2 > 0:
        print(f"Speedup vs PCRE2:         {avg_speedup_pcre2:.1f}x")

    # Check for token mismatches with PCRE2
    mismatches = 0
    for r in results.get("results", []):
        gen_tokens = r.get("generated", {}).get("tokens", [])
        pcre2_tokens = r.get("pcre2", {}).get("tokens", [])
        if gen_tokens != pcre2_tokens and r.get("pcre2", {}).get("success", False):
            mismatches += 1

    if mismatches > 0:
        print(f"Token mismatches vs PCRE2: {mismatches}")


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
    if not generate_cpp(pcre_pattern, name, verbose):
        return False

    # Step 2: Build
    print("\n[2/3] Building benchmark executable...")
    if not build_cpp(name, verbose):
        return False

    # Step 3: Run benchmark with STL and PCRE patterns
    print("\n[3/3] Running benchmark...")
    results = run_benchmark(name, stl_pattern, pcre_pattern, test_strings, verbose)
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
