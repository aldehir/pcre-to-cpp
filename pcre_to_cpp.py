#!/usr/bin/env python3
"""
PCRE to C++ Converter

Converts PCRE regular expressions into C++ functions that split input text
into chunks for LLM pretokenization.
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

    def _parse_alternation(self) -> Node:
        """Parse alternation: sequence ('|' sequence)*"""
        start_pos = self.pos
        alternatives = [self._parse_sequence()]

        while self._peek() == "|":
            self._advance()
            alternatives.append(self._parse_sequence())

        if len(alternatives) == 1:
            return alternatives[0]
        result = Alternation(alternatives)
        result.fragment = self.pattern[start_pos : self.pos]
        return result

    def _parse_sequence(self) -> Node:
        """Parse sequence: term+"""
        start_pos = self.pos
        terms = []

        while self.pos < self.length:
            # Stop at alternation or group end
            if self._peek() in ("|", ")") or self._peek() is None:
                break

            term = self._parse_term()
            if term is None:
                break
            terms.append(term)

        if len(terms) == 0:
            raise ValueError(f"Empty sequence at position {self.pos}")
        if len(terms) == 1:
            return terms[0]
        result = Sequence(terms)
        result.fragment = self.pattern[start_pos : self.pos]
        return result

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
            result = Quantifier(atom, min_c, max_c, greedy, possessive)
            result.fragment = self.pattern[start_pos : self.pos]
            return result

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
            result = AnyChar()
            result.fragment = self.pattern[start_pos : self.pos]
            return result

        # Anchors
        if ch == "^":
            start_pos = self.pos
            self._advance()
            result = Anchor("start")
            result.fragment = self.pattern[start_pos : self.pos]
            return result
        if ch == "$":
            start_pos = self.pos
            self._advance()
            result = Anchor("end")
            result.fragment = self.pattern[start_pos : self.pos]
            return result

        # Special characters that end atoms
        if ch in "|)":
            return None

        # Quantifiers shouldn't appear here
        if ch in "*+?{":
            return None

        # Literal character
        start_pos = self.pos
        self._advance()
        result = LiteralChar(ch)
        result.fragment = self.pattern[start_pos : self.pos]
        return result

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

            result = UnicodeCategory(category, negated)
            result.fragment = self.pattern[start_pos : self.pos]
            return result

        # Predefined classes
        if ch in "sSwWdD":
            self._advance()
            result = Predefined(ch)
            result.fragment = self.pattern[start_pos : self.pos]
            return result

        # Special escapes
        escape_map = {"r": "\r", "n": "\n", "t": "\t"}
        if ch in escape_map:
            self._advance()
            result = SpecialChar(escape_map[ch])
            result.fragment = self.pattern[start_pos : self.pos]
            return result

        # Hex escape: \xNN
        if ch == "x":
            self._advance()
            hex_digits = self.pattern[self.pos : self.pos + 2]
            if len(hex_digits) != 2:
                raise ValueError(f"Invalid hex escape at position {self.pos}")
            self._advance(2)
            result = LiteralChar(chr(int(hex_digits, 16)))
            result.fragment = self.pattern[start_pos : self.pos]
            return result

        # Escaped literal (special chars)
        self._advance()
        result = LiteralChar(ch)
        result.fragment = self.pattern[start_pos : self.pos]
        return result

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
        result = CharClass(items, negated)
        result.fragment = self.pattern[start_pos : self.pos]
        return result

    def _parse_cc_item(self) -> Node:
        """Parse a single item in a character class."""
        ch = self._peek()

        if ch == "\\":
            return self._parse_escape()

        start_pos = self.pos
        self._advance()
        result = LiteralChar(ch)
        result.fragment = self.pattern[start_pos : self.pos]
        return result

    def _cc_item_char(self, item: Node) -> Optional[str]:
        """Extract character from char class item, if possible."""
        if isinstance(item, LiteralChar):
            return item.char
        if isinstance(item, SpecialChar):
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

            if modifier == ":":
                # Non-capturing: (?:...)
                self._advance()
                child = self._parse_alternation()
                self._expect(")")
                result = GroupNode(child, capturing=False)
                result.fragment = self.pattern[start_pos : self.pos]
                return result

            elif modifier == "i":
                # Case-insensitive: (?i:...)
                self._advance()
                self._expect(":")
                child = self._parse_alternation()
                self._expect(")")
                result = GroupNode(child, capturing=False, case_insensitive=True)
                result.fragment = self.pattern[start_pos : self.pos]
                return result

            elif modifier == "=":
                # Positive lookahead: (?=...)
                self._advance()
                child = self._parse_alternation()
                self._expect(")")
                result = Lookahead(child, positive=True)
                result.fragment = self.pattern[start_pos : self.pos]
                return result

            elif modifier == "!":
                # Negative lookahead: (?!...)
                self._advance()
                child = self._parse_alternation()
                self._expect(")")
                result = Lookahead(child, positive=False)
                result.fragment = self.pattern[start_pos : self.pos]
                return result

            elif modifier == "<":
                # Lookbehind - not supported
                raise ValueError(f"Lookbehind is not supported (position {self.pos})")

            else:
                raise ValueError(
                    f"Unknown group modifier '?{modifier}' at position {self.pos}"
                )

        # Capturing group
        child = self._parse_alternation()
        self._expect(")")
        result = GroupNode(child, capturing=True)
        result.fragment = self.pattern[start_pos : self.pos]
        return result

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

    def _emit(self, line: str = ""):
        """Emit a single line with current indentation."""
        if line:
            self.lines.append("    " * self.indent_level + line)
        else:
            self.lines.append("")

    def _emit_block(self, template: str, **kwargs):
        """Emit a multi-line template block with auto-dedent.

        Uses $var syntax for substitution (no brace escaping needed for C++).
        """
        lines = template.split("\n")
        if lines and not lines[-1].strip():
            lines = lines[:-1]
        code = textwrap.dedent("\n".join(lines))
        code = Template(code).safe_substitute(kwargs)
        for line in code.split("\n"):
            self._emit(line)

    @contextmanager
    def _block(self, open_line: str = "{", close_line: str = "}"):
        """Context manager for indented blocks with braces."""
        self._emit(open_line)
        self.indent_level += 1
        try:
            yield
        finally:
            self.indent_level -= 1
            self._emit(close_line)

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
        if self.pattern:
            self._emit_block(
                """\
                // Auto-generated by pcre_to_cpp.py
                // Do not edit manually
                //
                // Original PCRE pattern:
                //
                //   $pattern
                //

                #include "unicode.h"

                #include <string>
                #include <vector>
                #include <cstdint>
            """,
                pattern=self.pattern,
            )
        else:
            self._emit_block("""\
                // Auto-generated by pcre_to_cpp.py
                // Do not edit manually

                #include "unicode.h"

                #include <string>
                #include <vector>
                #include <cstdint>
            """)

        # Function documentation
        self._emit_block(
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
            self._emit_block("""\
                const std::string & text,
                const std::vector<size_t> & offsets
            """)

        with self._block(") {"):
            self._emit_block("""\
                std::vector<size_t> bpe_offsets;
                bpe_offsets.reserve(offsets.size());

                // Convert UTF-8 to codepoints for pattern matching
                const auto cpts = unicode_cpts_from_utf8(text);
            """)

            # Emit shared backtracking stack if needed
            if self.uses_backtracking:
                self._emit_block("""
                    // Pre-allocated backtracking stack for quantifier matching
                    // Uses a single vector with base-index tracking to avoid per-match allocations
                    std::vector<size_t> stack;
                    stack.reserve(cpts.size() * 2);
                """)

            self._emit_block("""
                size_t start = 0;

                // Process each chunk from the previous tokenization pass
            """)

            with self._block("for (auto offset : offsets) {"):
                self._emit_block("""\
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

                    // Helper: Try to match at current position using predicate
                    // Returns true and advances mpos if condition is met
                    auto _try_match = [&](size_t& mpos, bool& mflag, auto condition) -> bool {
                        if (!mflag) return false;
                        if (condition()) {
                            mpos++;
                            return true;
                        }
                        mflag = false;
                        return false;
                    };
                """)

                # Emit stack helpers if backtracking is needed
                if self.uses_backtracking:
                    self._emit_block("""
                        // Stack helpers for backtracking
                        auto _stack_mark = [&]() -> size_t { return stack.size(); };
                        auto _stack_push = [&](size_t p) { stack.push_back(p); };
                        auto _stack_count = [&](size_t base) -> size_t { return stack.size() - base; };
                        auto _stack_get = [&](size_t base, size_t idx) -> size_t { return stack[base + idx - 1]; };
                        auto _stack_restore = [&](size_t base) { stack.resize(base); };
                    """)

                self._emit_block("""
                    // =======================================================
                    // Main matching loop
                    // Try each alternative in order. First match wins.
                    // On match: emit token boundary and continue from new position.
                    // On no match: consume single character as fallback.
                    // =======================================================
                """)
                with self._block("for (size_t pos = offset_ini; pos < offset_end; ) {"):
                    self._generate_match(self.ast)
                    self._emit_block("""
                        // No alternative matched - emit single character as token
                        _add_token(++pos);
                    """)

            self._emit_block("""
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
        self._emit("")
        self._emit(f"// Alternative: {ast.fragment}")

        with self._block():
            self._emit_block("""\
                size_t match_pos = pos;
                bool matched = true;

            """)

            # Generate matching for this alternative
            if isinstance(ast, Sequence):
                self._generate_sequence_match(ast.children)
            else:
                self._generate_node_match(ast)

            self._emit_block("""
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
            self._emit(
                f"_try_match(match_pos, matched, [&]{{ return unicode_tolower(_get_cpt(match_pos)) == {char_lower}; }}); // {char_desc} (case-insensitive)"
            )
        else:
            self._emit(
                f"_try_match(match_pos, matched, [&]{{ return _get_cpt(match_pos) == {char_code}; }}); // {char_desc}"
            )

    def _generate_special_match(self, node: SpecialChar):
        char_code = ord(node.char)
        char_desc = self._char_description(node.char)
        self._emit(
            f"_try_match(match_pos, matched, [&]{{ return _get_cpt(match_pos) == {char_code}; }}); // {char_desc}"
        )

    def _generate_charclass_match(
        self, node: CharClass, case_insensitive: bool = False
    ):
        """Generate match for character class."""
        needs_cpt, needs_flags, cond = self._charclass_condition_inline(
            node, case_insensitive
        )
        ci_suffix = " (case-insensitive)" if case_insensitive else ""

        # Emit multi-line lambda for readability
        self._emit(f"// {node.fragment}{ci_suffix}")
        self._emit("_try_match(match_pos, matched, [&]{")
        with self._indent_block():
            if needs_cpt:
                self._emit("uint32_t c = _get_cpt(match_pos);")
            if needs_flags:
                self._emit("auto f = _get_flags(match_pos);")
            self._emit(f"return {cond};")
        self._emit("});")

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
                    cond = (
                        "unicode_cpt_is_han(c)"
                        if not item.negated
                        else "!unicode_cpt_is_han(c)"
                    )
                else:
                    cond = self._unicode_cat_flags_condition(item)
                conditions.append(cond)
            elif isinstance(item, Predefined):
                needs_flags = True
                needs_cpt = (
                    True  # Some predefined classes use cpt (e.g., \w checks for '_')
                )
                cond = self._predefined_flags_condition(item)
                conditions.append(cond)
            elif isinstance(item, str):
                needs_cpt = True
                if case_insensitive and item.isalpha():
                    conditions.append(f"unicode_tolower(c) == {ord(item.lower())}")
                else:
                    conditions.append(f"c == {ord(item)}")

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

    def _unicode_cat_flags_condition(self, node: UnicodeCategory) -> str:
        """Generate flags-based condition for Unicode category (uses 'f' variable)."""
        cat = node.category
        cat_map = {
            "L": "f.is_letter",
            "N": "f.is_number",
            "P": "f.is_punctuation",
            "S": "f.is_symbol",
            "M": "f.is_accent_mark",
            "Z": "f.is_separator",
            "C": "f.is_control",
            "Lu": "(f.is_letter && f.is_uppercase)",
            "Ll": "(f.is_letter && f.is_lowercase)",
            "Lt": "(f.is_letter && f.is_uppercase)",
            "Lm": "(f.is_letter && !f.is_uppercase && !f.is_lowercase)",
            "Lo": "(f.is_letter && !f.is_uppercase && !f.is_lowercase)",
            "Nd": "f.is_number",
            "Nl": "f.is_number",
            "No": "f.is_number",
            "Mn": "f.is_accent_mark",
            "Mc": "f.is_accent_mark",
            "Me": "f.is_accent_mark",
        }

        if cat in cat_map:
            cond = cat_map[cat]
        else:
            cond = f"unicode_cpt_is_{cat.lower()}(c)"

        if node.negated:
            return f"!({cond})"
        return cond

    def _predefined_flags_condition(self, node: Predefined) -> str:
        """Generate flags-based condition for predefined class (uses 'c' and 'f' variables)."""
        name = node.name
        conditions = {
            "s": "f.is_whitespace",
            "S": "(!f.is_whitespace && f.as_uint())",
            "d": "f.is_number",
            "D": "(!f.is_number && f.as_uint())",
            "w": "(f.is_letter || f.is_number || c == '_')",
            "W": "(!(f.is_letter || f.is_number || c == '_') && f.as_uint())",
        }
        return conditions.get(name, "false")

    def _generate_unicode_cat_match(self, node: UnicodeCategory):
        """Generate match for Unicode category."""
        cond = self._unicode_cat_condition_inline(node)
        self._emit(
            f"_try_match(match_pos, matched, [&]{{ return {cond}; }}); // {node.fragment}"
        )

    def _unicode_cat_condition_inline(self, node: UnicodeCategory) -> str:
        """Generate inline condition for Unicode category using match_pos."""
        cat = node.category
        negated = node.negated

        cat_map = {
            "L": "_get_flags(match_pos).is_letter",
            "N": "_get_flags(match_pos).is_number",
            "P": "_get_flags(match_pos).is_punctuation",
            "S": "_get_flags(match_pos).is_symbol",
            "M": "_get_flags(match_pos).is_accent_mark",
            "Z": "_get_flags(match_pos).is_separator",
            "C": "_get_flags(match_pos).is_control",
            "Lu": "(_get_flags(match_pos).is_letter && _get_flags(match_pos).is_uppercase)",
            "Ll": "(_get_flags(match_pos).is_letter && _get_flags(match_pos).is_lowercase)",
            "Lt": "(_get_flags(match_pos).is_letter && _get_flags(match_pos).is_uppercase)",
            "Lm": "(_get_flags(match_pos).is_letter && !_get_flags(match_pos).is_uppercase && !_get_flags(match_pos).is_lowercase)",
            "Lo": "(_get_flags(match_pos).is_letter && !_get_flags(match_pos).is_uppercase && !_get_flags(match_pos).is_lowercase)",
            "Nd": "_get_flags(match_pos).is_number",
            "Nl": "_get_flags(match_pos).is_number",
            "No": "_get_flags(match_pos).is_number",
            "Mn": "_get_flags(match_pos).is_accent_mark",
            "Mc": "_get_flags(match_pos).is_accent_mark",
            "Me": "_get_flags(match_pos).is_accent_mark",
            "Han": "unicode_cpt_is_han(_get_cpt(match_pos))",
        }

        if cat in cat_map:
            cond = cat_map[cat]
        else:
            cond = f"unicode_cpt_is_{cat.lower()}(_get_cpt(match_pos))"

        if negated:
            return f"!({cond})"
        return cond

    def _generate_predefined_match(self, node: Predefined):
        """Generate match for predefined class."""
        cond = self._predefined_condition_inline(node)
        self._emit(
            f"_try_match(match_pos, matched, [&]{{ return {cond}; }}); // \\{node.name}"
        )

    def _predefined_condition_inline(self, node: Predefined) -> str:
        """Generate inline condition for predefined class using match_pos."""
        name = node.name

        conditions = {
            "s": "_get_flags(match_pos).is_whitespace",
            "S": "(!_get_flags(match_pos).is_whitespace && _get_flags(match_pos).as_uint())",
            "d": "_get_flags(match_pos).is_number",
            "D": "(!_get_flags(match_pos).is_number && _get_flags(match_pos).as_uint())",
            "w": "(_get_flags(match_pos).is_letter || _get_flags(match_pos).is_number || _get_cpt(match_pos) == '_')",
            "W": "(!(_get_flags(match_pos).is_letter || _get_flags(match_pos).is_number || _get_cpt(match_pos) == '_') && _get_flags(match_pos).as_uint())",
        }

        return conditions.get(name, "false")

    def _generate_any_match(self):
        """Generate match for any character."""
        self._emit(
            "_try_match(match_pos, matched, [&]{ return _get_cpt(match_pos) != OUT_OF_RANGE; }); // ."
        )

    def _generate_quantifier_match(
        self, node: Quantifier, case_insensitive: bool = False
    ):
        """Generate match for quantifier."""
        min_c = node.min_count
        max_c = node.max_count

        if min_c == 0 and max_c == 1:
            # Optional: ?
            self._generate_optional_match(node.child, case_insensitive)
        elif min_c == 0 and max_c == -1:
            # Zero or more: *
            self._generate_star_match(node.child, case_insensitive)
        elif min_c == 1 and max_c == -1:
            # One or more: +
            self._generate_plus_match(node.child, case_insensitive)
        elif min_c == max_c:
            # Exact count: {n}
            self._generate_exact_match(node.child, min_c, case_insensitive)
        else:
            # Range: {n,m}
            self._generate_range_match(node.child, min_c, max_c, case_insensitive)

    def _generate_optional_match(self, child: Node, case_insensitive: bool = False):
        """Generate match for optional (?)."""
        self._emit("// Optional match")
        with self._block():
            self._emit_block("""\
                size_t save_pos = match_pos;
                bool save_matched = matched;

            """)
            self._generate_node_match(child, case_insensitive)
            self._emit_block("""
                if (!matched) {
                    match_pos = save_pos;
                    matched = save_matched;
                }
            """)

    def _generate_star_match(self, child: Node, case_insensitive: bool = False):
        """Generate match for zero or more (*)."""
        self._emit("// Zero or more")
        with self._block("while (matched) {"):
            self._emit("size_t save_pos = match_pos;\n")
            self._generate_node_match(child, case_insensitive)
            self._emit_block("""
                if (!matched || match_pos == save_pos) {
                    match_pos = save_pos;
                    matched = true;
                    break;
                }
            """)

    def _generate_plus_match(self, child: Node, case_insensitive: bool = False):
        """Generate match for one or more (+)."""
        self._emit("// One or more")
        with self._block():
            self._emit("size_t count = 0;")
            with self._block("while (matched) {"):
                self._emit("size_t save_pos = match_pos;\n")
                self._generate_node_match(child, case_insensitive)
                self._emit_block("""
                    if (!matched || match_pos == save_pos) {
                        match_pos = save_pos;
                        matched = (count > 0);
                        break;
                    }
                    count++;
                """)

    def _generate_exact_match(
        self, child: Node, exact_count: int, case_insensitive: bool = False
    ):
        """Generate match for exact count {n}."""
        self._emit(f"// Exact {exact_count} matches")
        with self._block():
            self._emit("size_t count = 0;")
            with self._block(f"while (matched && count < {exact_count}) {{"):
                self._generate_node_match(child, case_insensitive)
                self._emit("if (matched) count++;")
            self._emit(f"if (count < {exact_count}) matched = false;")

    def _generate_range_match(
        self, child: Node, min_c: int, max_c: int, case_insensitive: bool = False
    ):
        """Generate match for range {n,m}."""
        if max_c == -1:
            self._emit(f"// {min_c} or more matches")
            with self._block():
                self._emit("size_t count = 0;")
                with self._block("while (matched) {"):
                    self._emit("size_t save_pos = match_pos;\n")
                    self._generate_node_match(child, case_insensitive)
                    self._emit_block(
                        """\
                        if (!matched || match_pos == save_pos) {
                            match_pos = save_pos;
                            matched = (count >= $min_c);
                            break;
                        }
                        count++;
                    """,
                        min_c=min_c,
                    )
        else:
            self._emit(f"// {min_c} to {max_c} matches")
            with self._block():
                self._emit("size_t count = 0;")
                with self._block(f"while (matched && count < {max_c}) {{"):
                    self._emit("size_t save_pos = match_pos;\n")
                    self._generate_node_match(child, case_insensitive)
                    self._emit_block(
                        """\
                        if (!matched || match_pos == save_pos) {
                            match_pos = save_pos;
                            matched = (count >= $min_c);
                            break;
                        }
                        count++;
                    """,
                        min_c=min_c,
                    )
                self._emit(f"if (count < {min_c}) matched = false;")

    def _generate_group_match(self, node: GroupNode):
        """Generate match for group."""
        if node.case_insensitive:
            self._emit("// Case-insensitive group")
            self._generate_node_match(node.child, case_insensitive=True)
        else:
            # Just match the child
            self._generate_node_match(node.child)

    def _generate_lookahead_match(
        self, node: Lookahead, case_insensitive: bool = False
    ):
        """Generate lookahead assertion."""
        lookahead_type = "Positive" if node.positive else "Negative"
        self._emit(f"// {lookahead_type} lookahead")

        with self._block():
            self._emit_block("""
                size_t save_match_pos = match_pos;
                bool save_matched = matched;
            """)

            self._generate_node_match(node.child, case_insensitive)

            self._emit("bool lookahead_success = matched;")
            self._emit("match_pos = save_match_pos;")

            if node.positive:
                self._emit("matched = save_matched && lookahead_success;")
            else:
                self._emit("matched = save_matched && !lookahead_success;")

    def _ast_needs_backtracking(self, ast: Node) -> bool:
        """Pre-scan AST to determine if any part needs backtracking.

        Used to decide whether to emit the shared stack declaration.
        """
        if isinstance(ast, Alternation):
            return any(self._ast_needs_backtracking(alt) for alt in ast.alternatives)
        elif isinstance(ast, Sequence):
            return self._needs_backtracking(ast.children)
        elif isinstance(ast, GroupNode):
            return self._ast_needs_backtracking(ast.child)
        elif isinstance(ast, Quantifier):
            # A lone quantifier doesn't need backtracking unless inside a sequence
            return False
        return False

    def _contains_backtracking_quantifier(self, node: Node) -> bool:
        """Check if a node or its children contain non-possessive quantifiers."""
        if isinstance(node, Quantifier):
            # Possessive quantifiers don't need backtracking
            return not node.possessive
        if isinstance(node, Sequence):
            return any(self._contains_backtracking_quantifier(c) for c in node.children)
        if isinstance(node, GroupNode):
            return self._contains_backtracking_quantifier(node.child)
        if isinstance(node, Alternation):
            return any(
                self._contains_backtracking_quantifier(a) for a in node.alternatives
            )
        return False

    def _needs_backtracking(self, children: list) -> bool:
        """Determine if a sequence needs backtracking support.

        Returns True if there's a non-possessive quantifier followed by something that could fail.
        Possessive quantifiers never backtrack, so they don't trigger this.
        """
        has_backtracking_quantifier = False
        for i, child in enumerate(children):
            # Check if this child is a non-possessive quantifier
            is_backtracking_quant = (
                isinstance(child, Quantifier) and not child.possessive
            ) or (
                isinstance(child, GroupNode)
                and self._contains_backtracking_quantifier(child.child)
            )

            # Check BEFORE updating has_backtracking_quantifier
            if has_backtracking_quantifier:
                # Something after a non-possessive quantifier - might need backtracking
                if isinstance(
                    child,
                    (
                        Lookahead,
                        LiteralChar,
                        CharClass,
                        Predefined,
                        UnicodeCategory,
                        SpecialChar,
                        AnyChar,
                    ),
                ):
                    return True
                if isinstance(child, GroupNode):
                    return True
                if isinstance(child, Alternation):
                    return True
                if is_backtracking_quant:
                    return True  # Multiple backtracking quantifiers in sequence

            if is_backtracking_quant:
                has_backtracking_quantifier = True
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
        quantifier_indices = []
        for i, child in enumerate(children):
            # Only include non-possessive quantifiers in backtracking
            if isinstance(child, Quantifier) and not child.possessive:
                quantifier_indices.append(i)

        if not quantifier_indices:
            for child in children:
                self._generate_node_match(child, case_insensitive)
            return

        pattern_str = "".join(c.fragment for c in children[:5])
        if len(children) > 5:
            pattern_str += "..."

        num_quants = len(quantifier_indices)

        self._emit(
            f"// Sequence with backtracking ({num_quants} quantifiers): {pattern_str}"
        )
        with self._block():
            self._emit("bool seq_matched = false;")
            self._emit("size_t bt_base = _stack_mark();  // Save stack state")

            # Generate elements before first quantifier
            first_quant_idx = quantifier_indices[0]
            for i in range(first_quant_idx):
                self._generate_node_match(children[i], case_insensitive)
                self._emit("if (!matched) { _stack_restore(bt_base); }")
                self._emit("else {")
                self.indent_level += 1

            # Generate stack-based backtracking loops for each quantifier
            self._generate_stack_based_backtracking(
                children, quantifier_indices, 0, case_insensitive
            )

            # Close the nested if-else blocks
            for i in range(first_quant_idx):
                self.indent_level -= 1
                self._emit("}")

            self._emit("")
            self._emit("_stack_restore(bt_base);  // Restore stack state")
            self._emit("matched = seq_matched;")

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
        self._emit_block(
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

        self._emit()
        with self._block(loop_cond):
            self._emit_block("""\
                size_t save_pos = match_pos;
                matched = true;

            """)
            self._generate_node_match(quant.child, case_insensitive)
            self._emit_block("""
                if (matched && match_pos > save_pos) {
                    _stack_push(match_pos);
                } else {
                    match_pos = save_pos;
                    break;
                }
            """)

        self._emit("")
        self._emit(f"size_t {count_name} = _stack_count({base_name});")
        self._emit("")

        # Loop through positions - direction depends on greedy vs lazy
        if greedy:
            # Greedy: try longest match first, backtrack to shorter
            self._emit(
                f"// Try quantifier {quant_num} positions longest-first (greedy, min_count={min_c})"
            )
            loop_header = f"for (size_t i{quant_num} = {count_name}; i{quant_num} > {min_c}; i{quant_num}--) {{"
        else:
            # Lazy: try shortest match first, extend if needed
            self._emit(
                f"// Try quantifier {quant_num} positions shortest-first (lazy, min_count={min_c})"
            )
            loop_header = f"for (size_t i{quant_num} = {min_c} + 1; i{quant_num} <= {count_name}; i{quant_num}++) {{"
        with self._block(loop_header):
            self._emit(f"match_pos = _stack_get({base_name}, i{quant_num});")
            self._emit("matched = true;")

            # Find elements between this quantifier and the next (or end)
            next_quant_idx = (
                quant_indices[quant_num + 1]
                if quant_num + 1 < len(quant_indices)
                else len(children)
            )

            # Generate elements between quantifiers
            for i in range(quant_idx + 1, next_quant_idx):
                self._generate_node_match(children[i], case_insensitive)
                self._emit("if (!matched) continue;")
                self._emit("")

            # Either recurse to next quantifier or finish
            if quant_num + 1 < len(quant_indices):
                self._generate_stack_based_backtracking(
                    children, quant_indices, quant_num + 1, case_insensitive
                )
                # Truncate nested quantifier's positions before trying next outer position
                next_base = f"q{quant_num + 1}_base"
                self._emit("")
                self._emit(f"_stack_restore({next_base});")
                self._emit("if (seq_matched) break;")
            else:
                for i in range(next_quant_idx, len(children)):
                    self._generate_node_match(children[i], case_insensitive)
                    if i < len(children) - 1:
                        self._emit("if (!matched) continue;")
                        self._emit("")
                self._emit("if (matched) { seq_matched = true; break; }")

    def _generate_nested_alternation(
        self, node: Alternation, case_insensitive: bool = False
    ):
        """Generate nested alternation (within a sequence)."""
        self._emit("")
        self._emit("// Nested alternation")
        with self._block("if (matched) {"):
            self._emit_block("""\
                size_t alt_save = match_pos;
                bool alt_matched = false;

            """)

            for i, alt in enumerate(node.alternatives):
                if i > 0:
                    self._emit("")
                    self._emit("if (!alt_matched) {")
                    self.indent_level += 1
                    self._emit_block("""\
                        match_pos = alt_save;
                        matched = true;

                    """)

                if isinstance(alt, Sequence):
                    for child in alt.children:
                        self._generate_node_match(child, case_insensitive)
                else:
                    self._generate_node_match(alt, case_insensitive)

                self._emit("")
                self._emit("alt_matched |= matched;")

                if i > 0:
                    self.indent_level -= 1
                    self._emit("}")

            self._emit("")
            self._emit("matched = alt_matched;")

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
