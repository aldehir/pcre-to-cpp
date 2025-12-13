#!/usr/bin/env python3
"""
Test runner for PCRE to C++ converter.

Orchestrates:
1. Generating C++ code from PCRE patterns
2. Building the C++ test harness
3. Running tests comparing C++ output vs HuggingFace tokenizers library
"""

import argparse
import io
import json
import os
import shutil
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
BUILD_DIR = HARNESS_DIR / "build"
GENERATED_DIR = HARNESS_DIR / "generated"


def generate_cpp(pattern: str) -> bool:
    """Generate C++ code from PCRE pattern."""
    output_file = GENERATED_DIR / "pattern_split.cpp"
    GENERATED_DIR.mkdir(exist_ok=True)

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


def build_cpp() -> bool:
    """Build the C++ test harness."""
    BUILD_DIR.mkdir(exist_ok=True)

    # Clean CMake cache if pattern changed (force reconfigure)
    cache_file = BUILD_DIR / "CMakeCache.txt"
    if cache_file.exists():
        cache_file.unlink()

    # Configure
    result = subprocess.run(
        ["cmake", ".."],
        cwd=BUILD_DIR,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"CMake configure failed: {result.stderr}", file=sys.stderr)
        return False

    # Build
    result = subprocess.run(
        ["cmake", "--build", "."],
        cwd=BUILD_DIR,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}\n{result.stdout}", file=sys.stderr)
        return False

    print("Build successful")
    return True


def run_cpp_tests(test_strings: list[str]) -> list[list[str]] | None:
    """Run C++ test harness on test strings."""
    # Find executable (Windows vs Unix)
    exe_path = BUILD_DIR / "regex_test.exe"
    if not exe_path.exists():
        exe_path = BUILD_DIR / "regex_test"
    if not exe_path.exists():
        exe_path = BUILD_DIR / "Debug" / "regex_test.exe"
    if not exe_path.exists():
        exe_path = BUILD_DIR / "Release" / "regex_test.exe"
    if not exe_path.exists():
        print(f"Could not find regex_test executable in {BUILD_DIR}", file=sys.stderr)
        return None

    input_json = json.dumps({"strings": test_strings})

    result = subprocess.run(
        [str(exe_path)],
        input=input_json,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )

    if result.returncode != 0:
        print(f"C++ execution failed: {result.stderr}", file=sys.stderr)
        return None

    try:
        output = json.loads(result.stdout)
        return output["results"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to parse C++ output: {e}\nOutput: {result.stdout}", file=sys.stderr)
        return None


def run_tokenizers_tests(pattern: str, test_strings: list[str]) -> list[list[str]]:
    """Run HuggingFace tokenizers library with the pattern.

    Uses the tokenizers library's Regex and Split pretokenizer to match
    the same semantics as LLM tokenizers (like Llama3).
    """
    from tokenizers import Regex
    from tokenizers.pre_tokenizers import Split

    # Create a pretokenizer that finds all matches of our pattern
    # behavior="isolated" keeps matches as separate tokens
    # invert=False means we match the pattern (this is what Llama3 uses)
    pretok = Split(pattern=Regex(pattern), behavior="isolated", invert=False)

    results = []
    for s in test_strings:
        # pre_tokenize_str returns list of (token, (start, end)) tuples
        output = pretok.pre_tokenize_str(s)
        tokens = [tok for tok, _span in output]
        results.append(tokens)

    return results


def run_hf_tokenizer_tests(tokenizer_name: str, test_strings: list[str]) -> list[list[str]]:
    """Run pretokenization using a HuggingFace tokenizer.

    This loads an actual tokenizer (GPT-2, Llama, etc.) and uses its
    pretokenizer to split test strings.
    """
    from tokenizers import Tokenizer

    try:
        tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Failed to load tokenizer '{tokenizer_name}': {e}", file=sys.stderr)
        return []

    pre_tokenizer = tokenizer.pre_tokenizer
    if pre_tokenizer is None:
        print(f"Tokenizer '{tokenizer_name}' has no pretokenizer", file=sys.stderr)
        return []

    results = []
    for s in test_strings:
        output = pre_tokenizer.pre_tokenize_str(s)
        tokens = [tok for tok, _span in output]
        results.append(tokens)

    return results


def compare_results(cpp_results: list[list[str]], ref_results: list[list[str]],
                    test_strings: list[str], ref_name: str = "Reference") -> bool:
    """Compare C++ and reference results, report differences."""
    all_match = True

    for i, (cpp, ref, text) in enumerate(zip(cpp_results, ref_results, test_strings)):
        if cpp != ref:
            all_match = False
            print(f"\n=== MISMATCH on test {i} ===")
            print(f"Input: {text!r}")
            print(f"C++:       {cpp}")
            print(f"{ref_name}: {ref}")

            # Show detailed diff
            max_len = max(len(cpp), len(ref))
            for j in range(max_len):
                cpp_tok = cpp[j] if j < len(cpp) else "<missing>"
                ref_tok = ref[j] if j < len(ref) else "<missing>"
                if cpp_tok != ref_tok:
                    print(f"  Token {j}: C++={cpp_tok!r} vs {ref_name}={ref_tok!r}")

    return all_match


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
            # Expect top-level list
            if isinstance(data, list):
                all_tests.extend(data)
            else:
                print(f"Warning: Expected list in {input_file}, got {type(data)}", file=sys.stderr)
    return all_tests


def run_test(pattern: str, test_strings: list[str], verbose: bool = False,
             tokenizer: str = None) -> bool:
    """Run a complete test cycle for a pattern."""
    print(f"\n{'='*60}")
    print(f"Pattern: {pattern[:60]}{'...' if len(pattern) > 60 else ''}")
    print(f"Reference: tokenizers library (regex-based)")
    print(f"Test strings: {len(test_strings)}")
    print('='*60)

    # Step 1: Generate C++
    print("\n[1/4] Generating C++ code...")
    if not generate_cpp(pattern):
        return False

    # Step 2: Build
    print("\n[2/4] Building C++ code...")
    if not build_cpp():
        return False

    # Step 3: Run C++ tests
    print("\n[3/4] Running C++ tests...")
    cpp_results = run_cpp_tests(test_strings)
    if cpp_results is None:
        return False

    # Step 4: Run reference tests
    print("\n[4/4] Running reference tests...")
    ref_results = run_tokenizers_tests(pattern, test_strings)
    ref_name = "tokenizers"

    if not ref_results:
        print("Failed to get reference results")
        return False

    # Compare
    print("\n--- Results ---")
    all_match = compare_results(cpp_results, ref_results, test_strings, ref_name)

    if all_match:
        print(f"\nSUCCESS: All {len(test_strings)} tests passed!")
    else:
        print(f"\nFAILED: Some tests did not match")

    if verbose:
        print("\n--- Detailed Results ---")
        for i, (text, cpp, ref) in enumerate(zip(test_strings, cpp_results, ref_results)):
            status = "OK" if cpp == ref else "FAIL"
            print(f"[{status}] {text!r}")
            print(f"      C++: {cpp}")
            print(f"      Ref: {ref}")

    return all_match


def main():
    parser = argparse.ArgumentParser(
        description="Test PCRE to C++ converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config file format (YAML):
- name: test_name
  pattern: "\\\\p{L}+"
  inputs:
    - natural_language
    - code

Input files are loaded from test-harness/inputs/ (top-level YAML lists).

Example usage:
    python run_tests.py                     # Run all tests from tests.yaml
    python run_tests.py -c my_tests.yaml    # Run tests from specific file
    python run_tests.py -n gpt2             # Run only the 'gpt2' test
    python run_tests.py -v                  # Verbose output
"""
    )
    parser.add_argument("--config", "-c", default="tests.yaml",
                        help="YAML config file with test patterns (default: tests.yaml)")
    parser.add_argument("--name", "-n", help="Run only the test with this name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--list", "-l", action="store_true", help="List available tests")

    args = parser.parse_args()

    # Load config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Create a tests.yaml file or specify one with --config")
        sys.exit(1)

    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Expect top-level list of patterns
    if isinstance(config, list):
        patterns = config
    else:
        print(f"Error: Expected list in config file, got {type(config)}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        print(f"Available tests in {config_path}:")
        for p in patterns:
            name = p.get("name", "unnamed")
            pattern = p.get("pattern", "")[:40]
            inputs = ", ".join(p.get("inputs", []))
            print(f"  - {name}: inputs=[{inputs}], pattern: {pattern}...")
        sys.exit(0)

    # Filter by name if specified
    if args.name:
        patterns = [p for p in patterns if p.get("name") == args.name]
        if not patterns:
            print(f"Error: No test named '{args.name}' found")
            sys.exit(1)

    # Run tests
    all_success = True
    for test_case in patterns:
        pattern = test_case["pattern"]
        name = test_case.get("name", "unnamed")
        input_names = test_case.get("inputs", [])
        print(f"\n### Testing: {name} ###")
        test_strings = load_test_inputs(input_names)
        if not test_strings:
            print(f"Warning: No test strings loaded for {name}", file=sys.stderr)
            continue
        # Always use regex-based testing (no tokenizer parameter)
        if not run_test(pattern, test_strings, args.verbose, tokenizer=None):
            all_success = False

    if all_success:
        print(f"\n=== ALL {len(patterns)} TEST(S) PASSED ===")
    else:
        print(f"\n=== SOME TESTS FAILED ===")

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
