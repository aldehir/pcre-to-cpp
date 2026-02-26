# PCRE to C++ Converter

Converts PCRE regex patterns into standalone C++ split functions for LLM
pretokenization. The generated code is iterative (no stack overflow on large
inputs) and supports Unicode property classes (`\p{L}`, `\p{N}`, `\p{Lu}`,
etc.) that `std::regex` lacks.

## Generating Code

```bash
python pcre_to_cpp.py --pattern "PATTERN" --name NAME [--output FILE]
```

`--pattern` is the PCRE regex. `--name` sets the generated function name
(`<name>_regex_split`). Output goes to stdout if `--output` is omitted.

Example:

```bash
python pcre_to_cpp.py \
    --pattern "'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+" \
    --name gpt2 \
    --output gpt2_split.cpp
```

Pre-generated files for common tokenizers are in `examples/`.

## Running Tests

Tests compare generated C++ output against PCRE2 on real text from HuggingFace
datasets. Patterns are defined in `config.yaml`.

```bash
python run.py test              # all patterns
python run.py test -n gpt2      # single pattern
```

## Running Benchmarks

```bash
python run.py bench                     # all patterns
python run.py bench -n gpt2             # single pattern
python run.py bench --iterations 100    # more iterations (default: 50)
python run.py bench --rebuild           # force full rebuild
```

Requires: `g++`, `libpcre2-dev`, `libboost-regex-dev`, `pyyaml`, `datasets`.

## Benchmark Results

Dataset: wikitext-103-v1 (20k samples), 50 iterations. PCRE2 uses JIT
compilation. STL patterns use ASCII approximations since `std::regex` does not
support Unicode properties.

```
Pattern   | Generated | STL Regex |  Boost  |  PCRE2  | vs STL | vs Boost | vs PCRE2
----------|-----------|-----------|---------|---------|--------|----------|--------
gpt2      |   5744ms  |  55700ms  | 33970ms |  7367ms |  9.7x  |   5.9x   |  1.3x
llama3    |   4107ms  |  78074ms  | 39887ms |  5457ms | 19.0x  |   9.7x   |  1.3x
gpt4o     |   7507ms  |  62673ms  | 27446ms |  7185ms |  8.3x  |   3.7x   |  1.0x
```

Zero token mismatches against PCRE2 across all patterns.

## Documentation

See [design.md](design.md) for supported PCRE features, generated code
structure, and required C++ helpers.
