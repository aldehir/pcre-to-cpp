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
┌─────────────────┐     ┌────────────────────────────┐     ┌────────────────┐
│  PCRE Pattern   │ ──► │  Recursive Descent Parser  │ ──► │       AST      │
└─────────────────┘     └────────────────────────────┘     └────────┬───────┘
                                                                    │
                                                                    ▼
                                                           ┌────────────────┐
                                                           │  AST Optimizer │
                                                           └────────┬───────┘
                                                                    │
                                                                    ▼
                                                           ┌────────────────┐
                                                           │  C++ Emitter   │
                                                           └────────┬───────┘
                                                                    │
                                                                    ▼
                                                           ┌────────────────┐
                                                           │  C++ Function  │
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

## AST Optimizer

After parsing, the AST goes through an optimization phase that simplifies and improves the tree structure. The optimizer runs until a fixed point is reached (no more changes).

### Optimization Passes

#### 1. Sequence Flattening

Nested sequences are flattened and trivial groups are unwrapped:

```
Seq([Seq([a, b]), c])  →  Seq([a, b, c])
Group(child, non-capturing, no-flags)  →  child
```

This removes unnecessary nesting from the AST and simplifies code generation.

#### 2. Alternation to CharClass

Single-character alternations are converted to character classes:

```
a|b|c  →  [abc]
\r|\n|\t  →  [\r\n\t]
```

This produces more efficient matching code since character class checks can be combined into a single condition.

#### 3. Common Prefix Extraction

Common prefixes are factored out from alternations:

```
abc|abd  →  ab(c|d)
foo|foobar|foobaz  →  foo(|bar|baz)
```

This avoids redundant matching of shared prefixes across alternatives.

### Implementation Details

The optimizer uses a recursive transformation approach:

1. **Bottom-up transformation**: Children are transformed before parents
2. **Fixed-point iteration**: Transformations run until the AST stops changing
3. **Deep equality checking**: Uses `nodes_equal()` to detect when optimization is complete

```python
def optimize(self, ast: Node) -> Node:
    prev = None
    current = ast
    while not self._ast_equal(prev, current):
        prev = current
        current = self._transform(current)
    return current
```

### Example

Input pattern: `'s|'t|'re|'ve|'m|'ll|'d`

After optimization:
```
'(s|t|re|ve|m|ll|d)
```

The common `'` prefix is extracted, reducing redundant character comparisons.

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

    // Pre-allocated backtracking stack (only if pattern needs backtracking)
    std::vector<size_t> bt_stack;
    bt_stack.reserve(cpts.size() * 2);

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

### Quantifiers (Simple Case)

When a quantifier is not followed by something that could fail, it uses simple iterative loops:

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

### Quantifiers with Backtracking

When a quantifier is followed by a lookahead, literal, or another quantifier, the generator produces **iterative backtracking** code. This handles patterns like `\s+(?!\S)` and `\s*[\r\n]+` correctly.

**Why backtracking is needed:** For `\s+(?!\S)` on input `"  x"`:
1. `\s+` greedily matches both spaces
2. `(?!\S)` checks next char - it's `x` (non-whitespace) → fails
3. Must backtrack: `\s+` gives up one space
4. `(?!\S)` checks next char - it's ` ` (whitespace) → succeeds!

**Strategy:** Use a shared pre-allocated stack with base index tracking. Collect all possible match positions into `bt_stack`, then try from longest to shortest:

```cpp
// Sequence with backtracking: \s+(?!\S)
{
    bool seq_matched = false;
    size_t bt_base = bt_stack.size();  // Save stack state

    // Collect all positions for \s+ into shared stack
    size_t q0_base = bt_stack.size();
    bt_stack.push_back(match_pos);
    while (true) {
        size_t save_pos = match_pos;
        // Try to match \s
        if (flags.is_whitespace) { match_pos++; }
        else { matched = false; }

        if (matched && match_pos > save_pos) {
            bt_stack.push_back(match_pos);
        } else {
            match_pos = save_pos;
            break;
        }
    }
    size_t q0_count = bt_stack.size() - q0_base;

    // Try positions from longest to shortest
    for (size_t i0 = q0_count; i0 > 1; i0--) {  // >1 for min_count=1
        match_pos = bt_stack[q0_base + i0 - 1];
        matched = true;

        // Test lookahead (?!\S)
        // ... lookahead code ...

        if (matched) { seq_matched = true; break; }
    }

    bt_stack.resize(bt_base);  // Restore stack state (O(1), no deallocation)
    matched = seq_matched;
}
```

### Multi-Quantifier Backtracking

For patterns with multiple quantifiers like `\s*[\r\n]+`, nested loops handle all combinations using the shared stack:

```cpp
// Sequence with backtracking (2 quantifiers): \s*[\r\n]+
{
    bool seq_matched = false;
    size_t bt_base = bt_stack.size();

    // Collect positions for \s* into shared stack
    size_t q0_base = bt_stack.size();
    bt_stack.push_back(match_pos);
    // ... collect all positions into bt_stack ...
    size_t q0_count = bt_stack.size() - q0_base;

    // Try q0 positions (outer loop)
    for (size_t i0 = q0_count; i0 > 0; i0--) {  // >0 for min_count=0
        match_pos = bt_stack[q0_base + i0 - 1];

        // Collect positions for [\r\n]+ from this point
        size_t q1_base = bt_stack.size();
        bt_stack.push_back(match_pos);
        // ... collect positions into bt_stack ...
        size_t q1_count = bt_stack.size() - q1_base;

        // Try q1 positions (inner loop)
        for (size_t i1 = q1_count; i1 > 1; i1--) {  // >1 for min_count=1
            match_pos = bt_stack[q1_base + i1 - 1];
            matched = true;

            if (matched) { seq_matched = true; break; }
        }
        bt_stack.resize(q1_base);  // Clean up q1's positions
        if (seq_matched) break;
    }

    bt_stack.resize(bt_base);  // Restore stack state
    matched = seq_matched;
}
```

This ensures patterns like `\s*[\r\n]+` correctly match `\n\n` by backtracking `\s*` to let `[\r\n]+` consume the newlines.

**Benefits of shared stack approach:**
- **One allocation** per function call instead of per-quantifier
- **No nested allocations** inside backtracking loops
- **O(1) cleanup** via `resize()` (truncates without deallocating)
- **Cache-friendly** memory access pattern

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

### Implemented Features

1. **Iterative backtracking**: Full support for quantifiers followed by lookaheads or other patterns
2. **Multi-quantifier backtracking**: Nested loops for patterns like `\s*[\r\n]+`
3. **Stack-safe**: All backtracking uses a single pre-allocated vector, not call stack recursion
4. **Memory-efficient**: Single `bt_stack` per function call, pre-sized to `2 * input_length`
5. **Zero per-quantifier allocations**: Uses base index tracking instead of separate vectors
6. **AST optimization**: Fixed-point optimizer with sequence flattening, alternation-to-charclass conversion, and common prefix extraction

### Potential Improvements

1. **String comparison**: Combine adjacent literal matches into string comparisons
2. **DFA compilation**: Convert simple patterns to state machines
3. **More scripts**: Add `\p{Hiragana}`, `\p{Katakana}`, etc.
4. **Lookbehind**: Could be added for fixed-width patterns
5. **Lazy quantifiers**: Currently parsed but not fully optimized

## File Structure

```
pcre-to-cpp/
├── pcre_to_cpp.py      # Main converter script
├── design.md           # This document
├── run_tests.py        # Test runner script
├── tests.yaml          # Test patterns and test input references
├── test-harness/
│   ├── CMakeLists.txt  # Build configuration
│   ├── test-main.cpp   # Test harness entry point
│   ├── unicode.cpp     # Unicode helper implementations
│   ├── unicode.h       # Unicode types and flags
│   ├── generated/      # Generated C++ pattern code
│   ├── inputs/         # Test input files (edge_cases.yaml, etc.)
│   └── build/          # Build artifacts
└── examples/           # Example patterns and reference code
```
