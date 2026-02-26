#!/usr/bin/env python3
"""
PCRE to C++ Converter

Converts PCRE regular expressions into C++ functions that split input text into
chunks for LLM pretokenization.

# Code Generation Strategy

Each regex construct maps to specific C++ code patterns. The generated function
iterates through codepoints, trying each alternative at each position until one
matches, then emits a token boundary.

## Literals and Simple Matchers

LiteralChar('a')
    TRY_MATCH(_get_cpt(match_pos) == 97);  // U+0061 'a'

SpecialChar('\n')
    TRY_MATCH(_get_cpt(match_pos) == 10);  // U+000A '\n'

AnyChar (.)
    TRY_MATCH(_get_cpt(match_pos) != OUT_OF_RANGE);  // .

## Character Classes

CharClass [a-z0-9]
    if (matched) {
        uint32_t c = _get_cpt(match_pos);
        matched = ((c >= 97 && c <= 122) || (c >= 48 && c <= 57));
        if (matched) { match_pos++; }
    }

CharClass (negated) [^a-z]
    if (matched) {
        uint32_t c = _get_cpt(match_pos);
        matched = (c != OUT_OF_RANGE && !(c >= 97 && c <= 122));
        if (matched) { match_pos++; }
    }

## Unicode Categories

\\p{L} (Letter)
    TRY_MATCH(_get_flags(match_pos).is_letter);

\\p{N} (Number)
    TRY_MATCH(_get_flags(match_pos).is_number);

\\P{L} (NOT Letter)
    TRY_MATCH(!(_get_flags(match_pos).is_letter));

\\p{Han} (special-cased)
    TRY_MATCH(unicode_cpt_is_han(_get_cpt(match_pos)));

## Predefined Classes

\\s (whitespace)
    TRY_MATCH(_get_flags(match_pos).is_whitespace);

\\d (digit)
    TRY_MATCH(_get_flags(match_pos).is_number);

\\w (word char)
    TRY_MATCH(_get_flags(match_pos).is_letter ||
              _get_flags(match_pos).is_number ||
              _get_cpt(match_pos) == '_');

\\S, \\D, \\W (negated)
    Include check for valid codepoint: (!condition && flags.as_uint())

## Quantifiers (Simple Cases)

? (optional)
    size_t save_pos = match_pos;
    bool save_matched = matched;
    <match child>
    if (!matched) { match_pos = save_pos; matched = save_matched; }

* (zero or more)
    while (matched) {
        size_t save_pos = match_pos;
        <match child>
        if (!matched || match_pos == save_pos) {
            match_pos = save_pos; matched = true; break;
        }
    }

+ (one or more)
    size_t count = 0;
    while (matched) {
        size_t save_pos = match_pos;
        <match child>
        if (!matched || match_pos == save_pos) {
            match_pos = save_pos; matched = (count > 0); break;
        }
        count++;
    }

{n} (exact count)
    size_t count = 0;
    while (matched && count < n) { <match child>; if (matched) count++; }
    if (count < n) matched = false;

{n,m} (range)
    Similar to above with min/max bounds

## Quantifiers with Backtracking

When a quantifier is followed by something that might fail (lookahead, literal,
etc.), backtracking is needed. Uses a shared stack with base-index tracking:

\\s+(?!\\S)  (whitespace not followed by non-whitespace)
    // Collect all possible match lengths into stack
    size_t q0_base = _stack_mark();
    _stack_push(match_pos);  // 0 matches
    while (true) {
        <try match \\s>
        if (matched && match_pos > save_pos) {
            _stack_push(match_pos);  // 1, 2, 3... matches
        } else break;
    }

    // Try longest first (greedy), backtrack to shorter
    for (size_t i0 = q0_count; i0 > min_count; i0--) {
        match_pos = _stack_get(q0_base, i0);
        matched = true;
        <check lookahead>
        if (matched) { seq_matched = true; break; }
    }

Lazy quantifiers (*?, +?) iterate shortest-first instead.
Possessive quantifiers (*+, ++) never backtrack - matched greedily, no stack.

## Alternation

a|bb|ccc
    // Alternative: a
    { match_pos = pos; matched = true; <match 'a'>
      if (matched && match_pos > pos) { pos = match_pos; _add_token(pos); continue; }
    }
    // Alternative: bb
    { match_pos = pos; matched = true; <match 'bb'>
      if (matched && match_pos > pos) { pos = match_pos; _add_token(pos); continue; }
    }
    // Alternative: ccc
    { ... }
    // No match - consume single character
    _add_token(++pos);

## Groups

(?:...)  (non-capturing)
    Just matches child, no special handling

(?i:...) (case-insensitive)
    Propagates case_insensitive=True to child matchers
    Literals use unicode_tolower() for comparison

## Lookahead

(?=pattern)  (positive lookahead)
    size_t save_match_pos = match_pos;
    bool save_matched = matched;
    <match pattern>
    bool lookahead_success = matched;
    match_pos = save_match_pos;  // restore position (zero-width)
    matched = save_matched && lookahead_success;

(?!pattern)  (negative lookahead)
    Same as above but: matched = save_matched && !lookahead_success;

## Anchors

^ (start), $ (end)
    No-ops in splitting context (pattern matches anywhere in chunk)
"""

import argparse
import sys
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass
from string import Template
from typing import Union, Tuple, Optional

# =============================================================================
# AST Node Types
# =============================================================================


@dataclass
class LiteralChar:
    """A literal character to match."""

    char: str
    fragment: str = ""

    def __repr__(self):
        return f"Literal({self.char!r})"


@dataclass
class CharClass:
    """A character class like [a-z] or [^0-9]."""

    items: list  # List of chars, (start, end) tuples, or nested nodes
    negated: bool = False
    fragment: str = ""

    def __repr__(self):
        neg = "^" if self.negated else ""
        return f"CharClass({neg}{self.items})"


@dataclass
class UnicodeCategory:
    """Unicode category like \\p{L} or \\P{N}."""

    category: str  # "L", "N", "P", "S", "M", "Han", etc.
    negated: bool = False
    fragment: str = ""

    def __repr__(self):
        p = "P" if self.negated else "p"
        return f"\\{p}{{{self.category}}}"


@dataclass
class Predefined:
    """Predefined character class like \\s, \\d, \\w."""

    name: str  # "s", "S", "d", "D", "w", "W"
    fragment: str = ""

    def __repr__(self):
        return f"\\{self.name}"


@dataclass
class SpecialChar:
    """Special character like \\r, \\n, \\t."""

    char: str  # The actual character value
    fragment: str = ""

    def __repr__(self):
        return f"Special({self.char!r})"


@dataclass
class AnyChar:
    """Matches any character (.)"""

    fragment: str = ""


@dataclass
class Quantifier:
    """Quantifier applied to a node."""

    child: "Node"
    min_count: int
    max_count: int  # -1 means unlimited
    greedy: bool = True
    possessive: bool = False  # If True, never backtrack
    fragment: str = ""

    def __repr__(self):
        if self.min_count == 0 and self.max_count == 1:
            q = "?"
        elif self.min_count == 0 and self.max_count == -1:
            q = "*"
        elif self.min_count == 1 and self.max_count == -1:
            q = "+"
        elif self.min_count == self.max_count:
            q = f"{{{self.min_count}}}"
        elif self.max_count == -1:
            q = f"{{{self.min_count},}}"
        else:
            q = f"{{{self.min_count},{self.max_count}}}"
        # Add modifier suffix
        if self.possessive:
            q += "+"
        elif not self.greedy:
            q += "?"
        return f"Quantifier({self.child}, {q})"


@dataclass
class Alternation:
    """Alternation of patterns (a|b|c)."""

    alternatives: list
    fragment: str = ""

    def __repr__(self):
        return f"Alt({self.alternatives})"


@dataclass
class Sequence:
    """Sequence of patterns (abc)."""

    children: list
    fragment: str = ""

    def __repr__(self):
        return f"Seq({self.children})"


@dataclass
class GroupNode:
    """A group (capturing or non-capturing)."""

    child: "Node"
    capturing: bool = False
    case_insensitive: bool = False
    fragment: str = ""

    def __repr__(self):
        flags = []
        if not self.capturing:
            flags.append("?:")
        if self.case_insensitive:
            flags.append("i")
        return f"Group({''.join(flags)}{self.child})"


@dataclass
class Lookahead:
    """Lookahead assertion (?=...) or (?!...)."""

    child: "Node"
    positive: bool  # True = (?=...), False = (?!...)
    fragment: str = ""

    def __repr__(self):
        op = "=" if self.positive else "!"
        return f"Lookahead({op}{self.child})"


@dataclass
class Anchor:
    """Anchor like ^ or $."""

    type: str  # "start", "end"
    fragment: str = ""

    def __repr__(self):
        return f"Anchor({self.type})"


# Type alias for all node types
Node = Union[
    LiteralChar,
    CharClass,
    UnicodeCategory,
    Predefined,
    SpecialChar,
    AnyChar,
    Quantifier,
    Alternation,
    Sequence,
    GroupNode,
    Lookahead,
    Anchor,
]


# =============================================================================
# Hand-written Recursive Descent Parser
# =============================================================================


class PCREParser:
    """
    Recursive descent parser for a subset of PCRE patterns.

    Grammar (roughly):
        pattern     -> alternation
        alternation -> sequence ('|' sequence)*
        sequence    -> term+
        term        -> atom quantifier?
        atom        -> literal | escape | charclass | group | '.'
        quantifier  -> '*' | '+' | '?' | '{n}' | '{n,}' | '{n,m}' ('?')?
        group       -> '(' pattern ')' | '(?:' pattern ')' | '(?i:' pattern ')'
                     | '(?=' pattern ')' | '(?!' pattern ')'
        charclass   -> '[' '^'? (cc_item)* ']'
        escape      -> '\\' (special | unicode_cat | predefined)
    """

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.pos = 0
        self.length = len(pattern)

    def parse(self) -> Node:
        result = self._parse_alternation()
        if self.pos < self.length:
            raise ValueError(
                f"Unexpected character at position {self.pos}: {self.pattern[self.pos]!r}"
            )
        return result

    def _peek(self, offset: int = 0) -> Optional[str]:
        pos = self.pos + offset
        if pos < self.length:
            return self.pattern[pos]
        return None

    def _advance(self, count: int = 1):
        self.pos += count

    def _match(self, s: str) -> bool:
        if self.pattern[self.pos : self.pos + len(s)] == s:
            self.pos += len(s)
            return True
        return False

    def _expect(self, s: str):
        if not self._match(s):
            raise ValueError(f"Expected {s!r} at position {self.pos}")

    def _with_fragment(self, node: Node, start_pos: int) -> Node:
        """Set fragment from pattern slice and return node."""
        node.fragment = self.pattern[start_pos : self.pos]
        return node

    def _parse_alternation(self) -> Node:
        """Parse alternation: sequence ('|' sequence)*"""
        start_pos = self.pos
        alternatives = [self._parse_sequence()]

        while self._peek() == "|":
            self._advance()
            alternatives.append(self._parse_sequence())

        if len(alternatives) == 1:
            return alternatives[0]
        return self._with_fragment(Alternation(alternatives), start_pos)

    def _parse_sequence(self) -> Node:
        """Parse sequence: term+"""
        start_pos = self.pos
        terms = []

        while self.pos < self.length:
            # Stop at alternation or group end
            if self._peek() in ("|", ")"):
                break

            term = self._parse_term()
            if term is None:
                break
            terms.append(term)

        if len(terms) == 0:
            raise ValueError(f"Empty sequence at position {self.pos}")
        if len(terms) == 1:
            return terms[0]
        return self._with_fragment(Sequence(terms), start_pos)

    def _parse_term(self) -> Optional[Node]:
        """Parse term: atom quantifier?"""
        start_pos = self.pos
        atom = self._parse_atom()
        if atom is None:
            return None

        # Check for quantifier
        quantifier = self._parse_quantifier()
        if quantifier:
            min_c, max_c, greedy, possessive = quantifier
            return self._with_fragment(Quantifier(atom, min_c, max_c, greedy, possessive), start_pos)

        return atom

    def _parse_atom(self) -> Optional[Node]:
        """Parse atom: literal | escape | charclass | group | '.' | anchor"""
        ch = self._peek()

        if ch is None:
            return None

        # Character class
        if ch == "[":
            return self._parse_charclass()

        # Group
        if ch == "(":
            return self._parse_group()

        # Escape sequence
        if ch == "\\":
            return self._parse_escape()

        # Any character
        if ch == ".":
            start_pos = self.pos
            self._advance()
            return self._with_fragment(AnyChar(), start_pos)

        # Anchors
        if ch == "^":
            start_pos = self.pos
            self._advance()
            return self._with_fragment(Anchor("start"), start_pos)
        if ch == "$":
            start_pos = self.pos
            self._advance()
            return self._with_fragment(Anchor("end"), start_pos)

        # Special characters that end atoms
        if ch in "|)":
            return None

        # Quantifiers shouldn't appear here
        if ch in "*+?{":
            return None

        # Literal character
        start_pos = self.pos
        self._advance()
        return self._with_fragment(LiteralChar(ch), start_pos)

    def _parse_escape(self) -> Node:
        """Parse escape sequence."""
        start_pos = self.pos
        self._expect("\\")
        ch = self._peek()

        if ch is None:
            raise ValueError("Unexpected end of pattern after backslash")

        # Unicode category: \p{...} or \P{...}
        if ch in "pP":
            negated = ch == "P"
            self._advance()
            self._expect("{")

            # Read category name
            cat_start = self.pos
            while self._peek() and self._peek() != "}":
                self._advance()
            category = self.pattern[cat_start : self.pos]
            self._expect("}")

            return self._with_fragment(UnicodeCategory(category, negated), start_pos)

        # Predefined classes
        if ch in "sSwWdD":
            self._advance()
            return self._with_fragment(Predefined(ch), start_pos)

        # Special escapes
        escape_map = {"r": "\r", "n": "\n", "t": "\t"}
        if ch in escape_map:
            self._advance()
            return self._with_fragment(SpecialChar(escape_map[ch]), start_pos)

        # Hex escape: \xNN
        if ch == "x":
            self._advance()
            hex_digits = self.pattern[self.pos : self.pos + 2]
            if len(hex_digits) != 2:
                raise ValueError(f"Invalid hex escape at position {self.pos}")
            self._advance(2)
            return self._with_fragment(LiteralChar(chr(int(hex_digits, 16))), start_pos)

        # Escaped literal (special chars)
        self._advance()
        return self._with_fragment(LiteralChar(ch), start_pos)

    def _parse_charclass(self) -> CharClass:
        """Parse character class: [...]"""
        start_pos = self.pos
        self._expect("[")

        negated = False
        if self._peek() == "^":
            negated = True
            self._advance()

        items = []

        while self._peek() and self._peek() != "]":
            item = self._parse_cc_item()

            # Check for range
            if self._peek() == "-" and self._peek(1) not in ("]", None):
                self._advance()  # consume '-'
                end_item = self._parse_cc_item()

                # Extract character values for range
                start_char = self._cc_item_char(item)
                end_char = self._cc_item_char(end_item)

                if start_char and end_char:
                    items.append((start_char, end_char))
                else:
                    # Can't make a range, add separately
                    items.append(item)
                    items.append(LiteralChar("-", fragment="-"))
                    items.append(end_item)
            else:
                items.append(item)

        self._expect("]")
        return self._with_fragment(CharClass(items, negated), start_pos)

    def _parse_cc_item(self) -> Node:
        """Parse a single item in a character class."""
        ch = self._peek()

        if ch == "\\":
            return self._parse_escape()

        start_pos = self.pos
        self._advance()
        return self._with_fragment(LiteralChar(ch), start_pos)

    def _cc_item_char(self, item: Node) -> Optional[str]:
        """Extract character from char class item, if possible."""
        if isinstance(item, (LiteralChar, SpecialChar)):
            return item.char
        return None

    def _parse_group(self) -> Node:
        """Parse group: (...) with various modifiers."""
        start_pos = self.pos
        self._expect("(")

        # Check for special group types
        if self._peek() == "?":
            self._advance()
            modifier = self._peek()

            if modifier == "i":
                # Case-insensitive: (?i:...)
                self._advance()
                self._expect(":")
            elif modifier in ":=!":
                self._advance()
            elif modifier == "<":
                raise ValueError(f"Lookbehind is not supported (position {self.pos})")
            else:
                raise ValueError(
                    f"Unknown group modifier '?{modifier}' at position {self.pos}"
                )

            child = self._parse_alternation()
            self._expect(")")

            if modifier == ":":
                node = GroupNode(child, capturing=False)
            elif modifier == "i":
                node = GroupNode(child, capturing=False, case_insensitive=True)
            elif modifier == "=":
                node = Lookahead(child, positive=True)
            else:  # modifier == "!"
                node = Lookahead(child, positive=False)

            return self._with_fragment(node, start_pos)

        # Capturing group
        child = self._parse_alternation()
        self._expect(")")
        return self._with_fragment(GroupNode(child, capturing=True), start_pos)

    def _parse_quantifier(self) -> Optional[Tuple[int, int, bool, bool]]:
        """Parse quantifier: *, +, ?, {n}, {n,}, {n,m} with optional lazy (?) or possessive (+)"""
        ch = self._peek()

        if ch == "*":
            self._advance()
            greedy, possessive = self._parse_quantifier_modifier()
            return (0, -1, greedy, possessive)

        if ch == "+":
            self._advance()
            greedy, possessive = self._parse_quantifier_modifier()
            return (1, -1, greedy, possessive)

        if ch == "?":
            self._advance()
            greedy, possessive = self._parse_quantifier_modifier()
            return (0, 1, greedy, possessive)

        if ch == "{":
            self._advance()

            # Parse min
            min_start = self.pos
            while self._peek() and self._peek().isdigit():
                self._advance()
            min_val = int(self.pattern[min_start : self.pos])

            if self._peek() == "}":
                # {n}
                self._advance()
                greedy, possessive = self._parse_quantifier_modifier()
                return (min_val, min_val, greedy, possessive)

            if self._peek() == ",":
                self._advance()

                if self._peek() == "}":
                    # {n,}
                    self._advance()
                    greedy, possessive = self._parse_quantifier_modifier()
                    return (min_val, -1, greedy, possessive)

                # {n,m}
                max_start = self.pos
                while self._peek() and self._peek().isdigit():
                    self._advance()
                max_val = int(self.pattern[max_start : self.pos])

                self._expect("}")
                greedy, possessive = self._parse_quantifier_modifier()
                return (min_val, max_val, greedy, possessive)

            raise ValueError(f"Invalid quantifier at position {self.pos}")

        return None

    def _parse_quantifier_modifier(self) -> Tuple[bool, bool]:
        """Parse optional lazy (?) or possessive (+) modifier.

        Returns:
            Tuple of (greedy, possessive)
        """
        if self._peek() == "?":
            self._advance()
            return (False, False)  # lazy: not greedy, not possessive
        elif self._peek() == "+":
            self._advance()
            return (True, True)  # possessive: greedy, possessive
        return (True, False)  # default: greedy, not possessive


def parse_pcre(pattern: str) -> Node:
    """Parse a PCRE pattern into an AST."""
    parser = PCREParser(pattern)
    return parser.parse()


def generate_cpp(ast: Node, name: str, pattern: str = "") -> str:
    """Generate C++ code from an AST.

    Args:
        name: Function name suffix (e.g., 'gpt2' -> unicode_regex_split_gpt2)
        ast: The parsed and optimized AST
        pattern: Optional original pattern string for documentation
    """
    return CppEmitter(ast, name, pattern).generate()


# =============================================================================
# C++ Code Emitter
# =============================================================================


class CppEmitter:
    """Generates C++ code from a parsed PCRE AST."""

    def __init__(self, ast: Node, name: str = "custom", pattern: str = ""):
        self.ast = ast
        self.name = name
        self.pattern = pattern
        self.indent_level = 0
        self.lines = []
        self.uses_backtracking = False  # Set by _ast_needs_backtracking()

    # =========================================================================
    # Emit Infrastructure
    # =========================================================================

    def emit(self, text: str = "", **kwargs):
        """Emit text with current indentation.

        Works for:
        - emit()              -> blank line
        - emit("single line") -> indented single line
        - emit('''multi
                 line''')     -> indented block, preserving relative indent

        Uses $var syntax for substitution (no brace escaping needed for C++).
        """
        if not text:
            self.lines.append("")
            return

        # Dedent and substitute
        text = textwrap.dedent(text)
        if kwargs:
            text = Template(text).safe_substitute(kwargs)

        # Split into lines
        lines = text.split("\n")

        # Strip only the FIRST line if empty (allows """\ vs """ formatting)
        if lines and not lines[0].strip():
            lines.pop(0)

        # Strip trailing empty lines
        while lines and not lines[-1].strip():
            lines.pop()

        if not lines:
            return

        # Emit each line with current indent level
        # After dedent, any remaining leading whitespace is relative indentation to preserve
        base_indent = "    " * self.indent_level
        for line in lines:
            if line.strip():
                # line.lstrip() removes leading whitespace, so line[:-len(lstripped)] is the relative indent
                content = line.lstrip()
                relative_indent = line[: len(line) - len(content)]
                self.lines.append(base_indent + relative_indent + content)
            else:
                self.lines.append("")

    @contextmanager
    def _block(self, open_line: str = "{", close_line: str = "}"):
        """Context manager for indented blocks with braces."""
        self.emit(open_line)
        self.indent_level += 1
        try:
            yield
        finally:
            self.indent_level -= 1
            self.emit(close_line)

    @contextmanager
    def _indent_block(self):
        """Context manager for indentation without braces."""
        self.indent_level += 1
        try:
            yield
        finally:
            self.indent_level -= 1

    def generate(self) -> str:
        """Generate C++ code for the AST."""
        self.lines = []
        self.uses_backtracking = self._ast_needs_backtracking(self.ast)
        func_name = self.name

        # File header with original pattern
        self.emit("""\
            // Auto-generated by pcre_to_cpp.py
            // Do not edit manually
        """)
        if self.pattern:
            self.emit("""\
                //
                // Original PCRE pattern:
                //
                //   $pattern
                //
            """, pattern=self.pattern)
        self.emit("""
            #include "unicode.h"

            #include <string>
            #include <vector>
            #include <cstdint>
        """)

        # Helpful macros
        self.emit("""

            // Macro for match attempts
            #define TRY_MATCH(cond) do {\\
                if (matched) { \\
                    if (cond) { match_pos++; } \\
                    else { matched = false; } \\
                } } while (0)
        """)

        # Function documentation (blank line before doc comment)
        self.emit(
            """

            /**
             * Split text into tokens using the '$func_name' pattern.
             *
             * @param text     UTF-8 encoded input string
             * @param offsets  Chunk sizes from previous tokenization pass
             * @return         New chunk sizes after applying this pattern
             */
            std::vector<size_t> unicode_regex_split_$func_name(
        """,
            func_name=func_name,
        )

        with self._indent_block():
            self.emit("""\
                const std::string & text,
                const std::vector<size_t> & offsets
            """)

        with self._block(") {"):
            self.emit("""\
                std::vector<size_t> bpe_offsets;
                bpe_offsets.reserve(offsets.size());

                // Convert UTF-8 to codepoints for pattern matching
                const auto cpts = unicode_cpts_from_utf8(text);
            """)

            # Emit shared backtracking stack if needed
            if self.uses_backtracking:
                self.emit("""

                    // Pre-allocated backtracking stack for quantifier matching
                    // Uses a single vector with base-index tracking to avoid per-match allocations
                    std::vector<size_t> stack;
                    stack.reserve(cpts.size() * 2);
                """)

            self.emit("""

                size_t start = 0;

                // Process each chunk from the previous tokenization pass
            """)

            with self._block("for (auto offset : offsets) {"):
                self.emit("""\
                    const size_t offset_ini = start;
                    const size_t offset_end = start + offset;
                    start = offset_end;

                    // Sentinel value for out-of-bounds codepoint access
                    static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;

                    // Helper: Get codepoint at position (returns OUT_OF_RANGE if outside chunk)
                    auto _get_cpt = [&](const size_t pos) -> uint32_t {
                        return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
                    };

                    // Helper: Get Unicode flags for codepoint at position
                    auto _get_flags = [&](const size_t pos) -> unicode_cpt_flags {
                        return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};
                    };

                    // Helper: Emit a token from _prev_end to 'end'
                    size_t _prev_end = offset_ini;
                    auto _add_token = [&](const size_t end) -> size_t {
                        size_t len = end - _prev_end;
                        if (len > 0) {
                            bpe_offsets.push_back(len);
                        }
                        _prev_end = end;
                        return len;
                    };
                """)

                # Emit stack helpers if backtracking is needed
                if self.uses_backtracking:
                    self.emit("""

                        // Stack helpers for backtracking
                        auto _stack_mark = [&]() -> size_t { return stack.size(); };
                        auto _stack_push = [&](size_t p) { stack.push_back(p); };
                        auto _stack_count = [&](size_t base) -> size_t { return stack.size() - base; };
                        auto _stack_get = [&](size_t base, size_t idx) -> size_t { return stack[base + idx - 1]; };
                        auto _stack_restore = [&](size_t base) { stack.resize(base); };
                    """)

                self.emit("""

                    // =======================================================
                    // Main matching loop
                    // Try each alternative in order. First match wins.
                    // On match: emit token boundary and continue from new position.
                    // On no match: consume single character as fallback.
                    // =======================================================
                """)
                with self._block("for (size_t pos = offset_ini; pos < offset_end; ) {"):
                    self._generate_match(self.ast)
                    self.emit("""

                        // No alternative matched - emit single character as token
                        _add_token(++pos);
                    """)

            self.emit("""
                return bpe_offsets;
            """)

        return "\n".join(self.lines)

    def _generate_match(self, ast: Node):
        """Generate matching code for the AST."""
        if isinstance(ast, Alternation):
            for i, alt in enumerate(ast.alternatives):
                self._generate_alternative(alt, is_first=(i == 0))
        else:
            self._generate_alternative(ast, is_first=True)

    def _generate_alternative(self, ast: Node, is_first: bool):
        """Generate code for a single alternative."""
        self.emit()
        self.emit(f"// Alternative: {ast.fragment}")

        with self._block():
            self.emit("""\
                size_t match_pos = pos;
                bool matched = true;

            """)

            # Generate matching for this alternative
            if isinstance(ast, Sequence):
                self._generate_sequence_match(ast.children)
            else:
                self._generate_node_match(ast)

            self.emit("""
                if (matched && match_pos > pos) {
                    pos = match_pos;
                    _add_token(pos);
                    continue;
                }
            """)

    def _generate_node_match(self, node: Node, case_insensitive: bool = False):
        """Generate matching code for a single node."""
        if isinstance(node, LiteralChar):
            self._generate_literal_match(node, case_insensitive)
        elif isinstance(node, CharClass):
            self._generate_charclass_match(node, case_insensitive)
        elif isinstance(node, UnicodeCategory):
            self._generate_unicode_cat_match(node)
        elif isinstance(node, Predefined):
            self._generate_predefined_match(node)
        elif isinstance(node, SpecialChar):
            self._generate_special_match(node)
        elif isinstance(node, AnyChar):
            self._generate_any_match()
        elif isinstance(node, Quantifier):
            self._generate_quantifier_match(node, case_insensitive)
        elif isinstance(node, Sequence):
            self._generate_sequence_match(node.children, case_insensitive)
        elif isinstance(node, GroupNode):
            self._generate_group_match(node)
        elif isinstance(node, Lookahead):
            self._generate_lookahead_match(node, case_insensitive)
        elif isinstance(node, Alternation):
            self._generate_nested_alternation(node, case_insensitive)
        elif isinstance(node, Anchor):
            # Anchors are typically no-ops in this context
            pass
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    def _generate_literal_match(
        self, node: LiteralChar, case_insensitive: bool = False
    ):
        char_code = ord(node.char)
        char_desc = self._char_description(node.char)

        if case_insensitive and node.char.isalpha():
            char_lower = ord(node.char.lower())
            self.emit(
                f"TRY_MATCH(unicode_tolower(_get_cpt(match_pos)) == {char_lower}); // {char_desc} (case-insensitive)"
            )
        else:
            self.emit(f"TRY_MATCH(_get_cpt(match_pos) == {char_code}); // {char_desc}")

    def _generate_special_match(self, node: SpecialChar):
        char_code = ord(node.char)
        char_desc = self._char_description(node.char)
        self.emit(f"TRY_MATCH(_get_cpt(match_pos) == {char_code}); // {char_desc}")

    def _generate_charclass_match(
        self, node: CharClass, case_insensitive: bool = False
    ):
        """Generate match for character class."""
        needs_cpt, needs_flags, cond = self._charclass_condition_inline(
            node, case_insensitive
        )
        ci_suffix = " (case-insensitive)" if case_insensitive else ""

        # Emit multi-line lambda for readability
        self.emit()
        self.emit(f"// {node.fragment}{ci_suffix}")
        with self._block("if (matched) {"):
            if needs_cpt:
                self.emit("uint32_t c = _get_cpt(match_pos);")
            if needs_flags:
                self.emit("auto f = _get_flags(match_pos);")
            self.emit(f"matched = ({cond});")
            self.emit("if (matched) { match_pos++; }")

    def _charclass_condition_inline(
        self, node: CharClass, case_insensitive: bool = False
    ) -> tuple:
        """Generate inline condition for character class.

        Returns:
            Tuple of (needs_cpt, needs_flags, condition_string)
        """
        needs_cpt = False
        needs_flags = False
        conditions = []

        for item in node.items:
            if isinstance(item, tuple):
                # Range like ('a', 'z')
                start, end = item
                needs_cpt = True
                if case_insensitive and start.isalpha() and end.isalpha():
                    conditions.append(
                        f"(unicode_tolower(c) >= {ord(start.lower())} && unicode_tolower(c) <= {ord(end.lower())})"
                    )
                else:
                    conditions.append(f"(c >= {ord(start)} && c <= {ord(end)})")
            elif isinstance(item, LiteralChar):
                needs_cpt = True
                if case_insensitive and item.char.isalpha():
                    conditions.append(f"unicode_tolower(c) == {ord(item.char.lower())}")
                else:
                    conditions.append(f"c == {ord(item.char)}")
            elif isinstance(item, SpecialChar):
                needs_cpt = True
                conditions.append(f"c == {ord(item.char)}")
            elif isinstance(item, UnicodeCategory):
                needs_flags = True
                if item.category == "Han":
                    needs_cpt = True
                cond = self._unicode_cat_condition(item, "f", "c")
                conditions.append(cond)
            elif isinstance(item, Predefined):
                needs_flags = True
                needs_cpt = True
                cond = self._predefined_condition(item, "f", "c")
                conditions.append(cond)

        if not conditions:
            # Empty character class
            if node.negated:
                needs_cpt = True
                return (needs_cpt, needs_flags, "c != OUT_OF_RANGE")
            else:
                return (False, False, "false")

        # Join conditions with ||
        combined = " || ".join(conditions)

        if node.negated:
            needs_cpt = True  # Need to check OUT_OF_RANGE
            return (needs_cpt, needs_flags, f"c != OUT_OF_RANGE && !({combined})")
        else:
            return (needs_cpt, needs_flags, combined)

    def _unicode_cat_condition(self, node: UnicodeCategory, flags_acc: str, cpt_acc: str) -> str:
        """Generate condition for Unicode category.

        Args:
            flags_acc: Expression for flags access (e.g., 'f' or '_get_flags(match_pos)')
            cpt_acc: Expression for codepoint access (e.g., 'c' or '_get_cpt(match_pos)')
        """
        cat = node.category
        cat_map = {
            "L": f"{flags_acc}.is_letter",
            "N": f"{flags_acc}.is_number",
            "P": f"{flags_acc}.is_punctuation",
            "S": f"{flags_acc}.is_symbol",
            "M": f"{flags_acc}.is_accent_mark",
            "Z": f"{flags_acc}.is_separator",
            "C": f"{flags_acc}.is_control",
            "Lu": f"({flags_acc}.is_letter && {flags_acc}.is_uppercase)",
            "Ll": f"({flags_acc}.is_letter && {flags_acc}.is_lowercase)",
            "Lt": f"({flags_acc}.is_letter && {flags_acc}.is_uppercase)",
            "Lm": f"({flags_acc}.is_letter && !{flags_acc}.is_uppercase && !{flags_acc}.is_lowercase)",
            "Lo": f"({flags_acc}.is_letter && !{flags_acc}.is_uppercase && !{flags_acc}.is_lowercase)",
            "Nd": f"{flags_acc}.is_number",
            "Nl": f"{flags_acc}.is_number",
            "No": f"{flags_acc}.is_number",
            "Mn": f"{flags_acc}.is_accent_mark",
            "Mc": f"{flags_acc}.is_accent_mark",
            "Me": f"{flags_acc}.is_accent_mark",
            "Han": f"unicode_cpt_is_han({cpt_acc})",
        }

        if cat in cat_map:
            cond = cat_map[cat]
        else:
            cond = f"unicode_cpt_is_{cat.lower()}({cpt_acc})"

        if node.negated:
            return f"!({cond})"
        return cond

    def _predefined_condition(self, node: Predefined, flags_acc: str, cpt_acc: str) -> str:
        """Generate condition for predefined class."""
        name = node.name
        conditions = {
            "s": f"{flags_acc}.is_whitespace",
            "S": f"(!{flags_acc}.is_whitespace && {flags_acc}.as_uint())",
            "d": f"{flags_acc}.is_number",
            "D": f"(!{flags_acc}.is_number && {flags_acc}.as_uint())",
            "w": f"({flags_acc}.is_letter || {flags_acc}.is_number || {cpt_acc} == '_')",
            "W": f"(!({flags_acc}.is_letter || {flags_acc}.is_number || {cpt_acc} == '_') && {flags_acc}.as_uint())",
        }
        return conditions.get(name, "false")

    def _generate_unicode_cat_match(self, node: UnicodeCategory):
        """Generate match for Unicode category."""
        cond = self._unicode_cat_condition(node, "_get_flags(match_pos)", "_get_cpt(match_pos)")
        self.emit(f"TRY_MATCH({cond}); // {node.fragment}")

    def _generate_predefined_match(self, node: Predefined):
        """Generate match for predefined class."""
        cond = self._predefined_condition(node, "_get_flags(match_pos)", "_get_cpt(match_pos)")
        self.emit(f"TRY_MATCH({cond}); // \\{node.name}")

    def _generate_any_match(self):
        """Generate match for any character."""
        self.emit("TRY_MATCH(_get_cpt(match_pos) != OUT_OF_RANGE); // .")

    def _generate_quantifier_match(
        self, node: Quantifier, case_insensitive: bool = False
    ):
        """Generate match for quantifier."""
        min_c = node.min_count
        max_c = node.max_count

        if min_c == 0 and max_c == 1:
            self._generate_optional_match(node.child, case_insensitive)
        elif min_c == max_c:
            self._generate_exact_match(node.child, min_c, case_insensitive)
        else:
            self._generate_range_match(node.child, min_c, max_c, case_insensitive)

    def _generate_optional_match(self, child: Node, case_insensitive: bool = False):
        """Generate match for optional (?)."""
        self.emit()
        self.emit("// Optional match")
        with self._block():
            self.emit("""\
                size_t save_pos = match_pos;
                bool save_matched = matched;

            """)
            self._generate_node_match(child, case_insensitive)
            self.emit("""
                if (!matched) {
                    match_pos = save_pos;
                    matched = save_matched;
                }
            """)

    def _generate_exact_match(
        self, child: Node, exact_count: int, case_insensitive: bool = False
    ):
        """Generate match for exact count {n}."""
        self.emit()
        self.emit(f"// Exact {exact_count} matches")
        with self._block():
            self.emit("size_t count = 0;")
            with self._block(f"while (matched && count < {exact_count}) {{"):
                self._generate_node_match(child, case_insensitive)
                self.emit("if (matched) count++;")
            self.emit(f"if (count < {exact_count}) matched = false;")

    def _generate_range_match(
        self, child: Node, min_c: int, max_c: int, case_insensitive: bool = False
    ):
        """Generate match for *, +, {n,}, {n,m} quantifiers."""
        self.emit()
        if max_c == -1:
            if min_c == 0:
                self.emit("// Zero or more")
            elif min_c == 1:
                self.emit("// One or more")
            else:
                self.emit(f"// {min_c} or more matches")
        else:
            self.emit(f"// {min_c} to {max_c} matches")

        loop_cond = "while (matched) {" if max_c == -1 else f"while (matched && count < {max_c}) {{"
        exit_cond = "true" if min_c == 0 else f"(count >= {min_c})"

        with self._block():
            self.emit("size_t count = 0;")
            with self._block(loop_cond):
                self.emit("size_t save_pos = match_pos;\n")
                self._generate_node_match(child, case_insensitive)
                self.emit(
                    """\
                    if (!matched || match_pos == save_pos) {
                        match_pos = save_pos;
                        matched = $exit_cond;
                        break;
                    }
                    count++;
                """,
                    exit_cond=exit_cond,
                )
            if max_c != -1:
                self.emit(f"if (count < {min_c}) matched = false;")

    def _generate_group_match(self, node: GroupNode):
        """Generate match for group."""
        if node.case_insensitive:
            self.emit()
            self.emit("// Case-insensitive group")
            self._generate_node_match(node.child, case_insensitive=True)
        else:
            # Just match the child
            self._generate_node_match(node.child)

    def _generate_lookahead_match(
        self, node: Lookahead, case_insensitive: bool = False
    ):
        """Generate lookahead assertion."""
        lookahead_type = "Positive" if node.positive else "Negative"
        self.emit()
        self.emit(f"// {lookahead_type} lookahead")

        with self._block():
            self.emit("""
                size_t save_match_pos = match_pos;
                bool save_matched = matched;
            """)

            self._generate_node_match(node.child, case_insensitive)

            self.emit("bool lookahead_success = matched;")
            self.emit("match_pos = save_match_pos;")

            if node.positive:
                self.emit("matched = save_matched && lookahead_success;")
            else:
                self.emit("matched = save_matched && !lookahead_success;")

    def _ast_needs_backtracking(self, ast: Node) -> bool:
        """Pre-scan AST to determine if any part needs backtracking.

        Used to decide whether to emit the shared stack declaration.
        """
        if isinstance(ast, Alternation):
            return any(self._ast_needs_backtracking(alt) for alt in ast.alternatives)
        if isinstance(ast, Sequence):
            return self._needs_backtracking(ast.children)
        if isinstance(ast, GroupNode):
            return self._ast_needs_backtracking(ast.child)
        return False

    def _needs_backtracking(self, children: list) -> bool:
        """Determine if a sequence needs backtracking support.

        Returns True if there's a non-possessive quantifier followed by something
        that could fail (anything except an anchor).
        """
        has_quantifier = False
        for child in children:
            if has_quantifier and not isinstance(child, Anchor):
                return True
            if isinstance(child, Quantifier) and not child.possessive:
                has_quantifier = True
        return False

    def _generate_sequence_match(self, children: list, case_insensitive: bool = False):
        """Generate match for a sequence, with backtracking support when needed."""
        if self._needs_backtracking(children):
            self._generate_sequence_with_backtracking(children, case_insensitive)
        else:
            # Simple case - no backtracking needed
            for child in children:
                self._generate_node_match(child, case_insensitive)

    def _generate_sequence_with_backtracking(
        self, children: list, case_insensitive: bool = False
    ):
        """Generate sequence matching with stack-based backtracking.

        Strategy: Use shared stack with base index tracking.
        For each non-possessive quantifier, collect ALL possible match positions into stack.
        Use nested loops to try all combinations, longest first (greedy) or shortest first (lazy).
        Possessive quantifiers are matched greedily without backtracking.
        """
        quantifier_indices = [
            i for i, child in enumerate(children)
            if isinstance(child, Quantifier) and not child.possessive
        ]

        pattern_str = "".join(c.fragment for c in children[:5])
        if len(children) > 5:
            pattern_str += "..."

        num_quants = len(quantifier_indices)

        self.emit()
        self.emit(
            f"// Sequence with backtracking ({num_quants} quantifiers): {pattern_str}"
        )
        with self._block():
            self.emit("bool seq_matched = false;")
            self.emit("size_t bt_base = _stack_mark();  // Save stack state")

            # Generate elements before first quantifier
            first_quant_idx = quantifier_indices[0]
            for i in range(first_quant_idx):
                self._generate_node_match(children[i], case_insensitive)
                self.emit("if (!matched) { _stack_restore(bt_base); }")
                self.emit("else {")
                self.indent_level += 1

            # Generate stack-based backtracking loops for each quantifier
            self._generate_stack_based_backtracking(
                children, quantifier_indices, 0, case_insensitive
            )

            # Close the nested if-else blocks
            for i in range(first_quant_idx):
                self.indent_level -= 1
                self.emit("}")

            self.emit()
            self.emit("_stack_restore(bt_base);  // Restore stack state")
            self.emit("matched = seq_matched;")

    def _generate_stack_based_backtracking(
        self,
        children: list,
        quant_indices: list,
        quant_num: int,
        case_insensitive: bool,
    ):
        """Generate stack-based nested loops for multiple quantifiers.

        Uses shared stack with base index tracking instead of per-quantifier vectors.
        For each quantifier, we:
        1. Collect all possible match positions into stack
        2. Loop from longest to shortest (greedy) or shortest to longest (lazy)
        3. Inside the loop, either recurse to next quantifier or try remaining pattern
        """
        quant_idx = quant_indices[quant_num]
        quant = children[quant_idx]
        min_c = quant.min_count
        max_c = quant.max_count
        greedy = quant.greedy
        base_name = f"q{quant_num}_base"
        count_name = f"q{quant_num}_count"

        # Collect positions for this quantifier using shared stack
        self.emit(
            """

            // Quantifier $quant_num: $quant_pattern
            size_t $base_name = _stack_mark();
            _stack_push(match_pos);
        """,
            quant_num=quant_num,
            quant_pattern=quant.fragment,
            base_name=base_name,
        )

        if max_c == -1:
            loop_cond = "while (true) {"
        else:
            loop_cond = f"while (_stack_count({base_name}) <= {max_c}) {{"

        self.emit()
        with self._block(loop_cond):
            self.emit("""\
                size_t save_pos = match_pos;
                matched = true;

            """)
            self._generate_node_match(quant.child, case_insensitive)
            self.emit("""
                if (matched && match_pos > save_pos) {
                    _stack_push(match_pos);
                } else {
                    match_pos = save_pos;
                    break;
                }
            """)

        self.emit()
        self.emit(f"size_t {count_name} = _stack_count({base_name});")

        # Loop through positions - direction depends on greedy vs lazy
        if greedy:
            # Greedy: try longest match first, backtrack to shorter
            self.emit()
            self.emit(
                f"// Try quantifier {quant_num} positions longest-first (greedy, min_count={min_c})"
            )
            loop_header = f"for (size_t i{quant_num} = {count_name}; i{quant_num} > {min_c}; i{quant_num}--) {{"
        else:
            # Lazy: try shortest match first, extend if needed
            self.emit()
            self.emit(
                f"// Try quantifier {quant_num} positions shortest-first (lazy, min_count={min_c})"
            )
            loop_header = f"for (size_t i{quant_num} = {min_c} + 1; i{quant_num} <= {count_name}; i{quant_num}++) {{"
        with self._block(loop_header):
            self.emit(f"match_pos = _stack_get({base_name}, i{quant_num});")
            self.emit("matched = true;")

            # Find elements between this quantifier and the next (or end)
            next_quant_idx = (
                quant_indices[quant_num + 1]
                if quant_num + 1 < len(quant_indices)
                else len(children)
            )

            # Generate elements between quantifiers
            for i in range(quant_idx + 1, next_quant_idx):
                self._generate_node_match(children[i], case_insensitive)
                self.emit("if (!matched) continue;")
                self.emit()

            # Either recurse to next quantifier or finish
            if quant_num + 1 < len(quant_indices):
                self._generate_stack_based_backtracking(
                    children, quant_indices, quant_num + 1, case_insensitive
                )
                # Truncate nested quantifier's positions before trying next outer position
                next_base = f"q{quant_num + 1}_base"
                self.emit()
                self.emit(f"_stack_restore({next_base});")
                self.emit("if (seq_matched) break;")
            else:
                for i in range(next_quant_idx, len(children)):
                    self._generate_node_match(children[i], case_insensitive)
                    if i < len(children) - 1:
                        self.emit("if (!matched) continue;")
                        self.emit()
                self.emit("if (matched) { seq_matched = true; break; }")

    def _generate_nested_alternation(
        self, node: Alternation, case_insensitive: bool = False
    ):
        """Generate nested alternation (within a sequence)."""
        self.emit()
        self.emit("// Nested alternation")
        with self._block("if (matched) {"):
            self.emit("""\
                size_t alt_save = match_pos;
                bool alt_matched = false;

            """)

            for i, alt in enumerate(node.alternatives):
                if i > 0:
                    self.emit()
                    self.emit("if (!alt_matched) {")
                    self.indent_level += 1
                    self.emit("""\
                        match_pos = alt_save;
                        matched = true;

                    """)

                if isinstance(alt, Sequence):
                    for child in alt.children:
                        self._generate_node_match(child, case_insensitive)
                else:
                    self._generate_node_match(alt, case_insensitive)

                self.emit()
                self.emit("alt_matched |= matched;")

                if i > 0:
                    self.indent_level -= 1
                    self.emit("}")

            self.emit()
            self.emit("matched = alt_matched;")

    def _escape_char(self, c: str) -> str:
        """Escape character for C++ comment."""
        if c == "\r":
            return "\\r"
        elif c == "\n":
            return "\\n"
        elif c == "\t":
            return "\\t"
        elif c == "\\":
            return "\\\\"
        elif c == "'":
            return "\\'"
        return c

    def _char_description(self, c: str) -> str:
        """Generate a descriptive comment for a character code.

        Returns format like: U+0020 ' ' or U+000A '\\n'
        """
        code = ord(c)
        escaped = self._escape_char(c)
        return f"U+{code:04X} '{escaped}'"


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Convert PCRE patterns to C++ code")
    parser.add_argument(
        "--pattern", "-p", required=True, help="The PCRE pattern to convert"
    )
    parser.add_argument(
        "--name",
        "-n",
        required=True,
        help="Name for the generated function (e.g., 'gpt2', 'llama3')",
    )
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    args = parser.parse_args()

    # Parse the pattern
    try:
        ast = parse_pcre(args.pattern)
    except ValueError as e:
        print(f"Error parsing pattern: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate C++ code
    cpp_code = generate_cpp(ast, args.name, args.pattern)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(cpp_code)
        print(f"Generated C++ code written to {args.output}")
    else:
        print(cpp_code)


if __name__ == "__main__":
    main()
