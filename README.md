# PCRE to C++ Converter

Converts PCRE regex patterns into standalone C++ functions for LLM pretokenization. Generates iterative matching code that avoids `std::regex` stack overflow on large inputs.

```mermaid
flowchart TD
    A["PCRE Pattern"] --> B["Recursive Descent Parser"]
    B --> C["AST"]
    C --> D["AST Optimizer"]
    D --> E["C++ Emitter"]
    E --> F["C++ Function"]
```

## Usage

```bash
python pcre_to_cpp.py --pattern "PATTERN" --name "NAME" [--output FILE]
```

Example:
```bash
python pcre_to_cpp.py \
    --pattern "'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+" \
    --name "gpt2" \
    --output gpt2_split.cpp
```

## Running Tests

```bash
python run_tests.py -c tests.yaml -v
python run_tests.py -c tests.yaml -v -n pattern_name  # run specific test
```

## Documentation

See [design.md](design.md) for supported PCRE features, generated code structure, and required C++ helpers.
