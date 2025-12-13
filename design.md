# PCRE to C++ Converter - Design Document

## Overview

This tool converts PCRE (Perl Compatible Regular Expressions) patterns into standalone C++ functions for LLM pretokenization. Instead of using a regex runtime, it generates iterative matching code that avoids stack overflow issues common with `std::regex` on large inputs.

### Why Not Use a Regex Library?

1. **Stack overflow**: C++ `std::regex` uses recursive backtracking that overflows on large inputs
2. **Performance**: Generated code is specialized for each pattern, avoiding runtime interpretation
3. **Portability**: No external dependencies beyond the C++ STL
4. **Control**: Full visibility into the matching logic for debugging and optimization

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  PCRE Pattern   │ ──► │  Recursive       │ ──► │      AST        │
│    (string)     │     │  Descent Parser  │     │   (Python)      │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌────────────────┐
                                                 │  C++ Emitter   │
                                                 └────────┬───────┘
                                                          │
                                                          ▼
                                                 ┌────────────────┐
                                                 │  C++ Function  │
                                                 │  (iterative)   │
                                                 └────────────────┘
```

## Parser Design

The parser is a hand-written recursive descent parser. This gives us full control over error messages and makes the code easy to understand and modify.

### Grammar

```
pattern     → alternation
alternation → sequence ('|' sequence)*
sequence    → term+
term        → atom quantifier?
atom        → literal | escape | charclass | group | '.' | anchor
quantifier  → ('*' | '+' | '?' | '{n}' | '{n,}' | '{n,m}') '?'?
group       → '(' pattern ')'
            | '(?:' pattern ')'      # non-capturing
            | '(?i:' pattern ')'     # case-insensitive
            | '(?=' pattern ')'      # positive lookahead
            | '(?!' pattern ')'      # negative lookahead
charclass   → '[' '^'? cc_item* ']'
cc_item     → literal | escape | range
escape      → '\' (special | unicode_cat | predefined)
```

### AST Node Types

```python
LiteralChar(char)           # Single character: 'a', '1', etc.
CharClass(items, negated)   # Character class: [a-z], [^0-9]
UnicodeCategory(cat, neg)   # Unicode category: \p{L}, \P{N}
Predefined(name)            # Predefined class: \s, \d, \w
SpecialChar(char)           # Special escape: \r, \n, \t
AnyChar()                   # Dot: .
Quantifier(child, min, max) # Quantifiers: *, +, ?, {n,m}
Alternation(alternatives)   # Alternation: a|b|c
Sequence(children)          # Sequence: abc
GroupNode(child, flags)     # Groups: (...), (?:...), (?i:...)
Lookahead(child, positive)  # Lookahead: (?=...), (?!...)
Anchor(type)                # Anchors: ^, $
```

## Supported PCRE Features

### Fully Supported

| Feature | Syntax | Example |
|---------|--------|---------|
| Literals | any char | `a`, `1`, `@` |
| Escaped literals | `\char` | `\.`, `\[`, `\\` |
| Character classes | `[...]` | `[a-z]`, `[^0-9]` |
| Negated classes | `[^...]` | `[^\s]` |
| Ranges in classes | `a-z` | `[a-zA-Z0-9]` |
| Unicode categories | `\p{X}` | `\p{L}`, `\p{Han}` |
| Negated categories | `\P{X}` | `\P{N}` |
| Predefined classes | `\s \d \w` | `\s+`, `\d{3}` |
| Negated predefined | `\S \D \W` | `\S+` |
| Special escapes | `\r \n \t` | `[\r\n]` |
| Hex escapes | `\xNN` | `\x20` |
| Any character | `.` | `a.b` |
| Zero or more | `*` | `a*` |
| One or more | `+` | `a+` |
| Optional | `?` | `a?` |
| Exact count | `{n}` | `\d{3}` |
| Range count | `{n,m}` | `\d{1,3}` |
| At least n | `{n,}` | `\d{2,}` |
| Lazy quantifiers | `*? +? ??` | `.*?` |
| Alternation | `\|` | `cat\|dog` |
| Grouping | `(...)` | `(ab)+` |
| Non-capturing | `(?:...)` | `(?:ab)+` |
| Case-insensitive | `(?i:...)` | `(?i:'s\|'t)` |
| Positive lookahead | `(?=...)` | `\d(?=px)` |
| Negative lookahead | `(?!...)` | `\s+(?!\S)` |
| Anchors | `^ $` | `^start` |

### Not Supported (Will Error)

| Feature | Syntax | Reason |
|---------|--------|--------|
| Lookbehind | `(?<=...)` `(?<!...)` | Complex to implement iteratively |
| Backreferences | `\1`, `\2` | Requires capture tracking |
| Named groups | `(?P<name>...)` | Not needed for tokenization |
| Atomic groups | `(?>...)` | Rarely used |
| Possessive quantifiers | `*+`, `++` | Rarely used |
| Unicode scripts | `\p{Greek}` | Would need script tables |
| Recursion | `(?R)` | Not applicable |
| Conditionals | `(?(cond)...)` | Too complex |

## Unicode Category Support

### Major Categories

| Category | Description | C++ Condition |
|----------|-------------|---------------|
| `\p{L}` | Any letter | `flags.is_letter` |
| `\p{N}` | Any number | `flags.is_number` |
| `\p{P}` | Punctuation | `flags.is_punctuation` |
| `\p{S}` | Symbol | `flags.is_symbol` |
| `\p{M}` | Mark (combining) | `flags.is_accent_mark` |
| `\p{Z}` | Separator | `flags.is_separator` |
| `\p{C}` | Control/Other | `flags.is_control` |

### Letter Subcategories

| Category | Description | C++ Condition |
|----------|-------------|---------------|
| `\p{Lu}` | Uppercase letter | `is_letter && is_uppercase` |
| `\p{Ll}` | Lowercase letter | `is_letter && is_lowercase` |
| `\p{Lt}` | Titlecase letter | `is_letter && is_uppercase` (approx) |
| `\p{Lm}` | Modifier letter | `is_letter && !is_uppercase && !is_lowercase` |
| `\p{Lo}` | Other letter | `is_letter && !is_uppercase && !is_lowercase` |

### Number Subcategories

| Category | Description | C++ Condition |
|----------|-------------|---------------|
| `\p{Nd}` | Decimal digit | `flags.is_number` |
| `\p{Nl}` | Letter number | `flags.is_number` |
| `\p{No}` | Other number | `flags.is_number` |

### Mark Subcategories

| Category | Description | C++ Condition |
|----------|-------------|---------------|
| `\p{Mn}` | Non-spacing mark | `flags.is_accent_mark` |
| `\p{Mc}` | Spacing combining | `flags.is_accent_mark` |
| `\p{Me}` | Enclosing mark | `flags.is_accent_mark` |

### Script Categories

| Category | Description | C++ Condition |
|----------|-------------|---------------|
| `\p{Han}` | CJK Ideographs | `unicode_cpt_is_han(cpt)` |

For unrecognized categories like `\p{Foo}`, the generator emits `unicode_cpt_is_foo(cpt)` which you must implement.

## Generated Code Structure

Each pattern generates a function with this signature:

```cpp
static std::vector<size_t> unicode_regex_split_NAME(
    const std::string & text,
    const std::vector<size_t> & offsets
);
```

### Input/Output

- **Input `text`**: UTF-8 encoded input string
- **Input `offsets`**: Chunk sizes from previous pattern (or `{text.length()}` initially)
- **Output**: New chunk sizes after applying this pattern

### Generated Code Template

```cpp
static std::vector<size_t> unicode_regex_split_NAME(...) {
    std::vector<size_t> bpe_offsets;
    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        start = offset_end;

        // Helper lambdas
        auto _get_cpt = [&](size_t pos) -> uint32_t { ... };
        auto _get_flags = [&](size_t pos) -> unicode_cpt_flags { ... };
        auto _add_token = [&](size_t end) -> size_t { ... };

        // Main matching loop
        for (size_t pos = offset_ini; pos < offset_end; ) {
            // Try each alternative
            { /* Alternative 1 */ }
            { /* Alternative 2 */ }
            ...

            // Fallback: consume single character
            _add_token(++pos);
        }
    }
    return bpe_offsets;
}
```

## Required C++ Helpers

Your C++ code must provide these functions:

```cpp
// UTF-8 to codepoint conversion
std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8);

// Codepoint to UTF-8 (if needed for output)
std::string unicode_cpt_to_utf8(uint32_t cpt);

// Get flags for a codepoint
unicode_cpt_flags unicode_cpt_flags_from_cpt(uint32_t cpt);

// Case folding
uint32_t unicode_tolower(uint32_t cpt);

// Script detection (if using \p{Han})
bool unicode_cpt_is_han(uint32_t cpt);
```

### unicode_cpt_flags Structure

```cpp
struct unicode_cpt_flags {
    uint16_t is_undefined   : 1;
    uint16_t is_number      : 1;  // \p{N}
    uint16_t is_letter      : 1;  // \p{L}
    uint16_t is_separator   : 1;  // \p{Z}
    uint16_t is_accent_mark : 1;  // \p{M}
    uint16_t is_punctuation : 1;  // \p{P}
    uint16_t is_symbol      : 1;  // \p{S}
    uint16_t is_control     : 1;  // \p{C}
    uint16_t is_whitespace  : 1;  // \s
    uint16_t is_lowercase   : 1;
    uint16_t is_uppercase   : 1;
    uint16_t is_nfd         : 1;

    uint16_t as_uint() const;     // Check if defined
};
```

## Usage

### Command Line

```bash
python pcre_to_cpp.py --pattern "PATTERN" --name "NAME" [--output FILE]
```

### Examples

```bash
# GPT-2 tokenizer pattern
python pcre_to_cpp.py \
    --pattern "'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+" \
    --name "gpt2" \
    --output gpt2_split.cpp

# Llama3 tokenizer pattern (case-insensitive contractions)
python pcre_to_cpp.py \
    --pattern "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+" \
    --name "llama3" \
    --output llama3_split.cpp
```

## Matching Strategy

### Alternation

Each alternative is tried in order. First match wins:

```cpp
// Alternative 1: 's
{
    size_t match_pos = pos;
    bool matched = true;

    if (matched && _get_cpt(match_pos) == '\'') match_pos++;
    else matched = false;

    if (matched && _get_cpt(match_pos) == 's') match_pos++;
    else matched = false;

    if (matched && match_pos > pos) {
        pos = match_pos;
        _add_token(pos);
        continue;
    }
}
// Alternative 2: 't
{ ... }
```

### Quantifiers

Quantifiers use iterative loops, not recursion:

```cpp
// One or more: \p{L}+
{
    size_t count = 0;
    while (matched) {
        size_t save_pos = match_pos;
        // Try to match one \p{L}
        if (flags.is_letter) {
            match_pos++;
            count++;
        } else {
            match_pos = save_pos;
            matched = (count > 0);
            break;
        }
    }
}
```

### Lookahead

Lookahead saves/restores position:

```cpp
// Negative lookahead: (?!\S)
{
    size_t save_pos = match_pos;
    bool save_matched = matched;

    // Try to match \S
    if (!flags.is_whitespace && flags.as_uint()) {
        match_pos++;
    } else {
        matched = false;
    }

    bool lookahead_success = matched;
    match_pos = save_pos;  // Restore position
    matched = save_matched && !lookahead_success;  // Negate for (?!)
}
```

### Case-Insensitive Matching

Case-insensitive groups use `unicode_tolower()`:

```cpp
// (?i:'s) - matches 's, 'S
if (matched && unicode_tolower(_get_cpt(match_pos)) == 's') {
    match_pos++;
} else {
    matched = false;
}
```

## Limitations and Future Work

### Current Limitations

1. **No lookbehind**: Would require scanning backwards, complex to implement
2. **No backreferences**: Would need capture tracking infrastructure
3. **Approximate subcategories**: `\p{Lt}` (titlecase) approximated as uppercase
4. **No script categories**: Only `\p{Han}` is implemented

### Potential Improvements

1. **Optimization**: Combine adjacent literal matches into string comparisons
2. **DFA compilation**: Convert simple patterns to state machines
3. **More scripts**: Add `\p{Hiragana}`, `\p{Katakana}`, etc.
4. **Lookbehind**: Could be added for fixed-width patterns
5. **Testing**: Add test suite comparing output to Python `regex` module

## File Structure

```
pcre-to-cpp/
├── pcre_to_cpp.py      # Main converter script
├── design.md           # This document
├── plan.md             # Original planning notes
├── examples/
│   ├── unicode.cpp     # Reference implementation with helpers
│   ├── unicode.h       # Header with unicode_cpt_flags
│   └── tokenizers.cpp  # Example PCRE patterns from LLM tokenizers
└── staging/            # Build artifacts
```
