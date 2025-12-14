#!/usr/bin/env python3
"""
PCRE to C++ Converter

Converts PCRE regular expressions into C++ functions that split input text
into chunks for LLM pretokenization.
"""

import argparse
import re
import sys
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass
from string import Template
from typing import Union, List, Tuple, Optional

# =============================================================================
# AST Node Types
# =============================================================================

@dataclass
class LiteralChar:
    """A literal character to match."""
    char: str

    def __repr__(self):
        return f"Literal({self.char!r})"

@dataclass
class CharClass:
    """A character class like [a-z] or [^0-9]."""
    items: list  # List of chars, (start, end) tuples, or nested nodes
    negated: bool = False

    def __repr__(self):
        neg = "^" if self.negated else ""
        return f"CharClass({neg}{self.items})"

@dataclass
class UnicodeCategory:
    """Unicode category like \\p{L} or \\P{N}."""
    category: str  # "L", "N", "P", "S", "M", "Han", etc.
    negated: bool = False

    def __repr__(self):
        p = "P" if self.negated else "p"
        return f"\\{p}{{{self.category}}}"

@dataclass
class Predefined:
    """Predefined character class like \\s, \\d, \\w."""
    name: str  # "s", "S", "d", "D", "w", "W"

    def __repr__(self):
        return f"\\{self.name}"

@dataclass
class SpecialChar:
    """Special character like \\r, \\n, \\t."""
    char: str  # The actual character value

    def __repr__(self):
        return f"Special({self.char!r})"

@dataclass
class AnyChar:
    """Matches any character (.)"""
    pass

@dataclass
class Quantifier:
    """Quantifier applied to a node."""
    child: 'Node'
    min_count: int
    max_count: int  # -1 means unlimited
    greedy: bool = True
    possessive: bool = False  # If True, never backtrack

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

    def __repr__(self):
        return f"Alt({self.alternatives})"

@dataclass
class Sequence:
    """Sequence of patterns (abc)."""
    children: list

    def __repr__(self):
        return f"Seq({self.children})"

@dataclass
class GroupNode:
    """A group (capturing or non-capturing)."""
    child: 'Node'
    capturing: bool = False
    case_insensitive: bool = False

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
    child: 'Node'
    positive: bool  # True = (?=...), False = (?!...)

    def __repr__(self):
        op = "=" if self.positive else "!"
        return f"Lookahead({op}{self.child})"

@dataclass
class Anchor:
    """Anchor like ^ or $."""
    type: str  # "start", "end"

    def __repr__(self):
        return f"Anchor({self.type})"

# Type alias for all node types
Node = Union[
    LiteralChar, CharClass, UnicodeCategory, Predefined, SpecialChar,
    AnyChar, Quantifier, Alternation, Sequence, GroupNode, Lookahead, Anchor
]


def nodes_equal(a: Node, b: Node) -> bool:
    """Deep equality check for AST nodes."""
    if type(a) != type(b):
        return False
    if isinstance(a, LiteralChar):
        return a.char == b.char
    if isinstance(a, SpecialChar):
        return a.char == b.char
    if isinstance(a, Predefined):
        return a.name == b.name
    if isinstance(a, UnicodeCategory):
        return a.category == b.category and a.negated == b.negated
    if isinstance(a, AnyChar):
        return True
    if isinstance(a, Anchor):
        return a.type == b.type
    if isinstance(a, CharClass):
        if a.negated != b.negated or len(a.items) != len(b.items):
            return False
        for x, y in zip(a.items, b.items):
            # Items can be tuples (ranges) or nodes
            if isinstance(x, tuple) and isinstance(y, tuple):
                if x != y:
                    return False
            elif isinstance(x, tuple) or isinstance(y, tuple):
                return False
            elif not nodes_equal(x, y):
                return False
        return True
    if isinstance(a, Quantifier):
        return (a.min_count == b.min_count and
                a.max_count == b.max_count and
                a.greedy == b.greedy and
                a.possessive == b.possessive and
                nodes_equal(a.child, b.child))
    if isinstance(a, Sequence):
        if len(a.children) != len(b.children):
            return False
        return all(nodes_equal(x, y) for x, y in zip(a.children, b.children))
    if isinstance(a, Alternation):
        if len(a.alternatives) != len(b.alternatives):
            return False
        return all(nodes_equal(x, y) for x, y in zip(a.alternatives, b.alternatives))
    if isinstance(a, GroupNode):
        return (a.capturing == b.capturing and
                a.case_insensitive == b.case_insensitive and
                nodes_equal(a.child, b.child))
    if isinstance(a, Lookahead):
        return a.positive == b.positive and nodes_equal(a.child, b.child)
    return False


# =============================================================================
# AST Optimizer
# =============================================================================

class ASTOptimizer:
    """Optimizes regex AST before code generation."""

    def optimize(self, ast: Node) -> Node:
        """Apply all optimizations (run until fixed point)."""
        prev = None
        current = ast
        # Run until no changes (fixed point)
        while not self._ast_equal(prev, current):
            prev = current
            current = self._transform(current)
        return current

    def _ast_equal(self, a: Optional[Node], b: Optional[Node]) -> bool:
        """Check if two ASTs are equal (handles None)."""
        if a is None or b is None:
            return a is b
        return nodes_equal(a, b)

    def _transform(self, node: Node) -> Node:
        """Recursively transform a node and its children."""
        # First transform children
        node = self._transform_children(node)
        # Then apply optimizations to this node
        node = self._flatten_sequence(node)
        node = self._alternation_to_charclass(node)
        node = self._extract_common_prefix(node)
        return node

    def _transform_children(self, node: Node) -> Node:
        """Recursively transform children of a node."""
        if isinstance(node, (LiteralChar, SpecialChar, Predefined,
                             UnicodeCategory, AnyChar, Anchor)):
            return node  # Leaf nodes

        if isinstance(node, CharClass):
            # Transform items that are nodes (not tuples for ranges)
            new_items = []
            for item in node.items:
                if isinstance(item, tuple):
                    new_items.append(item)
                else:
                    new_items.append(self._transform(item))
            return CharClass(new_items, node.negated)

        if isinstance(node, Quantifier):
            return Quantifier(self._transform(node.child),
                              node.min_count, node.max_count, node.greedy, node.possessive)

        if isinstance(node, Sequence):
            return Sequence([self._transform(c) for c in node.children])

        if isinstance(node, Alternation):
            return Alternation([self._transform(a) for a in node.alternatives])

        if isinstance(node, GroupNode):
            return GroupNode(self._transform(node.child),
                             node.capturing, node.case_insensitive)

        if isinstance(node, Lookahead):
            return Lookahead(self._transform(node.child), node.positive)

        return node

    def _flatten_sequence(self, node: Node) -> Node:
        """Flatten nested sequences and unwrap trivial groups."""
        if isinstance(node, Sequence):
            flattened = []
            for child in node.children:
                if isinstance(child, Sequence):
                    flattened.extend(child.children)
                else:
                    flattened.append(child)
            if len(flattened) == 0:
                return Sequence([])
            if len(flattened) == 1:
                return flattened[0]
            return Sequence(flattened)

        # Unwrap non-capturing, non-case-insensitive groups
        if isinstance(node, GroupNode):
            if not node.capturing and not node.case_insensitive:
                return node.child

        return node

    def _is_single_char(self, node: Node) -> bool:
        """Check if node matches exactly one character."""
        return isinstance(node, (LiteralChar, SpecialChar, Predefined, UnicodeCategory))

    def _alternation_to_charclass(self, node: Node) -> Node:
        """Convert alternation of single chars to CharClass."""
        if not isinstance(node, Alternation):
            return node

        # Check if ALL alternatives are single-char items
        if not all(self._is_single_char(alt) for alt in node.alternatives):
            return node

        return CharClass(items=list(node.alternatives), negated=False)

    def _to_sequence(self, node: Node) -> Sequence:
        """Wrap non-sequence nodes in a Sequence."""
        if isinstance(node, Sequence):
            return node
        return Sequence([node])

    def _extract_common_prefix(self, node: Node) -> Node:
        """Extract common prefix from alternation."""
        if not isinstance(node, Alternation):
            return node

        if len(node.alternatives) < 2:
            return node

        # Convert all alternatives to sequences
        seqs = [self._to_sequence(alt) for alt in node.alternatives]

        # Find common prefix length
        prefix_len = 0
        while True:
            # Check if all sequences have enough elements
            if not all(len(s.children) > prefix_len for s in seqs):
                break
            # Check if all elements at this position are equal
            first = seqs[0].children[prefix_len]
            if not all(nodes_equal(s.children[prefix_len], first) for s in seqs[1:]):
                break
            prefix_len += 1

        if prefix_len == 0:
            return node

        # Build: prefix + Alternation(suffixes)
        prefix = list(seqs[0].children[:prefix_len])
        suffixes = []
        for s in seqs:
            remaining = s.children[prefix_len:]
            if len(remaining) == 0:
                suffixes.append(Sequence([]))  # Empty match
            elif len(remaining) == 1:
                suffixes.append(remaining[0])
            else:
                suffixes.append(Sequence(list(remaining)))

        result_children = prefix + [Alternation(suffixes)]
        if len(result_children) == 1:
            return result_children[0]
        return Sequence(result_children)


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
            raise ValueError(f"Unexpected character at position {self.pos}: {self.pattern[self.pos]!r}")
        return result

    def _peek(self, offset: int = 0) -> Optional[str]:
        pos = self.pos + offset
        if pos < self.length:
            return self.pattern[pos]
        return None

    def _advance(self, count: int = 1):
        self.pos += count

    def _match(self, s: str) -> bool:
        if self.pattern[self.pos:self.pos + len(s)] == s:
            self.pos += len(s)
            return True
        return False

    def _expect(self, s: str):
        if not self._match(s):
            raise ValueError(f"Expected {s!r} at position {self.pos}")

    def _parse_alternation(self) -> Node:
        """Parse alternation: sequence ('|' sequence)*"""
        alternatives = [self._parse_sequence()]

        while self._peek() == '|':
            self._advance()
            alternatives.append(self._parse_sequence())

        if len(alternatives) == 1:
            return alternatives[0]
        return Alternation(alternatives)

    def _parse_sequence(self) -> Node:
        """Parse sequence: term+"""
        terms = []

        while self.pos < self.length:
            # Stop at alternation or group end
            if self._peek() in ('|', ')') or self._peek() is None:
                break

            term = self._parse_term()
            if term is None:
                break
            terms.append(term)

        if len(terms) == 0:
            raise ValueError(f"Empty sequence at position {self.pos}")
        if len(terms) == 1:
            return terms[0]
        return Sequence(terms)

    def _parse_term(self) -> Optional[Node]:
        """Parse term: atom quantifier?"""
        atom = self._parse_atom()
        if atom is None:
            return None

        # Check for quantifier
        quantifier = self._parse_quantifier()
        if quantifier:
            min_c, max_c, greedy, possessive = quantifier
            return Quantifier(atom, min_c, max_c, greedy, possessive)

        return atom

    def _parse_atom(self) -> Optional[Node]:
        """Parse atom: literal | escape | charclass | group | '.' | anchor"""
        ch = self._peek()

        if ch is None:
            return None

        # Character class
        if ch == '[':
            return self._parse_charclass()

        # Group
        if ch == '(':
            return self._parse_group()

        # Escape sequence
        if ch == '\\':
            return self._parse_escape()

        # Any character
        if ch == '.':
            self._advance()
            return AnyChar()

        # Anchors
        if ch == '^':
            self._advance()
            return Anchor("start")
        if ch == '$':
            self._advance()
            return Anchor("end")

        # Special characters that end atoms
        if ch in '|)':
            return None

        # Quantifiers shouldn't appear here
        if ch in '*+?{':
            return None

        # Literal character
        self._advance()
        return LiteralChar(ch)

    def _parse_escape(self) -> Node:
        """Parse escape sequence."""
        self._expect('\\')
        ch = self._peek()

        if ch is None:
            raise ValueError("Unexpected end of pattern after backslash")

        # Unicode category: \p{...} or \P{...}
        if ch in 'pP':
            negated = (ch == 'P')
            self._advance()
            self._expect('{')

            # Read category name
            cat_start = self.pos
            while self._peek() and self._peek() != '}':
                self._advance()
            category = self.pattern[cat_start:self.pos]
            self._expect('}')

            return UnicodeCategory(category, negated)

        # Predefined classes
        if ch in 'sSwWdD':
            self._advance()
            return Predefined(ch)

        # Special escapes
        escape_map = {'r': '\r', 'n': '\n', 't': '\t'}
        if ch in escape_map:
            self._advance()
            return SpecialChar(escape_map[ch])

        # Hex escape: \xNN
        if ch == 'x':
            self._advance()
            hex_digits = self.pattern[self.pos:self.pos + 2]
            if len(hex_digits) != 2:
                raise ValueError(f"Invalid hex escape at position {self.pos}")
            self._advance(2)
            return LiteralChar(chr(int(hex_digits, 16)))

        # Escaped literal (special chars)
        self._advance()
        return LiteralChar(ch)

    def _parse_charclass(self) -> CharClass:
        """Parse character class: [...]"""
        self._expect('[')

        negated = False
        if self._peek() == '^':
            negated = True
            self._advance()

        items = []

        while self._peek() and self._peek() != ']':
            item = self._parse_cc_item()

            # Check for range
            if self._peek() == '-' and self._peek(1) not in (']', None):
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
                    items.append(LiteralChar('-'))
                    items.append(end_item)
            else:
                items.append(item)

        self._expect(']')
        return CharClass(items, negated)

    def _parse_cc_item(self) -> Node:
        """Parse a single item in a character class."""
        ch = self._peek()

        if ch == '\\':
            return self._parse_escape()

        self._advance()
        return LiteralChar(ch)

    def _cc_item_char(self, item: Node) -> Optional[str]:
        """Extract character from char class item, if possible."""
        if isinstance(item, LiteralChar):
            return item.char
        if isinstance(item, SpecialChar):
            return item.char
        return None

    def _parse_group(self) -> Node:
        """Parse group: (...) with various modifiers."""
        self._expect('(')

        # Check for special group types
        if self._peek() == '?':
            self._advance()
            modifier = self._peek()

            if modifier == ':':
                # Non-capturing: (?:...)
                self._advance()
                child = self._parse_alternation()
                self._expect(')')
                return GroupNode(child, capturing=False)

            elif modifier == 'i':
                # Case-insensitive: (?i:...)
                self._advance()
                self._expect(':')
                child = self._parse_alternation()
                self._expect(')')
                return GroupNode(child, capturing=False, case_insensitive=True)

            elif modifier == '=':
                # Positive lookahead: (?=...)
                self._advance()
                child = self._parse_alternation()
                self._expect(')')
                return Lookahead(child, positive=True)

            elif modifier == '!':
                # Negative lookahead: (?!...)
                self._advance()
                child = self._parse_alternation()
                self._expect(')')
                return Lookahead(child, positive=False)

            elif modifier == '<':
                # Lookbehind - not supported
                raise ValueError(f"Lookbehind is not supported (position {self.pos})")

            else:
                raise ValueError(f"Unknown group modifier '?{modifier}' at position {self.pos}")

        # Capturing group
        child = self._parse_alternation()
        self._expect(')')
        return GroupNode(child, capturing=True)

    def _parse_quantifier(self) -> Optional[Tuple[int, int, bool, bool]]:
        """Parse quantifier: *, +, ?, {n}, {n,}, {n,m} with optional lazy (?) or possessive (+)"""
        ch = self._peek()

        if ch == '*':
            self._advance()
            greedy, possessive = self._parse_quantifier_modifier()
            return (0, -1, greedy, possessive)

        if ch == '+':
            self._advance()
            greedy, possessive = self._parse_quantifier_modifier()
            return (1, -1, greedy, possessive)

        if ch == '?':
            self._advance()
            greedy, possessive = self._parse_quantifier_modifier()
            return (0, 1, greedy, possessive)

        if ch == '{':
            self._advance()

            # Parse min
            min_start = self.pos
            while self._peek() and self._peek().isdigit():
                self._advance()
            min_val = int(self.pattern[min_start:self.pos])

            if self._peek() == '}':
                # {n}
                self._advance()
                greedy, possessive = self._parse_quantifier_modifier()
                return (min_val, min_val, greedy, possessive)

            if self._peek() == ',':
                self._advance()

                if self._peek() == '}':
                    # {n,}
                    self._advance()
                    greedy, possessive = self._parse_quantifier_modifier()
                    return (min_val, -1, greedy, possessive)

                # {n,m}
                max_start = self.pos
                while self._peek() and self._peek().isdigit():
                    self._advance()
                max_val = int(self.pattern[max_start:self.pos])

                self._expect('}')
                greedy, possessive = self._parse_quantifier_modifier()
                return (min_val, max_val, greedy, possessive)

            raise ValueError(f"Invalid quantifier at position {self.pos}")

        return None

    def _parse_quantifier_modifier(self) -> Tuple[bool, bool]:
        """Parse optional lazy (?) or possessive (+) modifier.

        Returns:
            Tuple of (greedy, possessive)
        """
        if self._peek() == '?':
            self._advance()
            return (False, False)  # lazy: not greedy, not possessive
        elif self._peek() == '+':
            self._advance()
            return (True, True)  # possessive: greedy, possessive
        return (True, False)  # default: greedy, not possessive


def parse_pcre(pattern: str) -> Node:
    """Parse a PCRE pattern into an AST."""
    parser = PCREParser(pattern)
    return parser.parse()

# =============================================================================
# C++ Code Emitter
# =============================================================================

class CppEmitter:
    """Generates C++ code from a parsed PCRE AST."""

    def __init__(self, function_name: str):
        self.function_name = function_name
        self.indent_level = 0
        self.lines = []
        self.temp_counter = 0
        self.required_helpers = set()
        self.uses_backtracking = False  # Set by _ast_needs_backtracking()

    # =========================================================================
    # Emit Infrastructure
    # =========================================================================

    def _emit(self, line: str = ""):
        """Emit a single line with current indentation."""
        self.lines.append("    " * self.indent_level + line)

    def _emit_block(self, template: str, vars: dict = None):
        """Emit a multi-line template block with auto-dedent.

        Uses $var syntax for substitution (no brace escaping needed for C++).
        Pass locals() as vars for convenience.
        """
        code = textwrap.dedent(template).strip()
        if vars:
            code = Template(code).safe_substitute(vars)
        for line in code.split('\n'):
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

    def _temp_var(self, prefix: str = "tmp") -> str:
        self.temp_counter += 1
        return f"{prefix}_{self.temp_counter}"

    def generate(self, ast: Node, pattern: str = None) -> str:
        """Generate C++ code for the given AST.

        Args:
            ast: The parsed AST node
            pattern: Optional original pattern string for documentation
        """
        self.lines = []
        self.required_helpers = set()
        self.uses_backtracking = self._ast_needs_backtracking(ast)
        func_name = self.function_name

        # File header with original pattern
        self._emit("// Auto-generated by pcre_to_cpp.py")
        self._emit("// Do not edit manually")
        if pattern:
            self._emit("//")
            self._emit("// Original PCRE pattern:")
            self._emit("//")
            self._emit(f"//   {pattern}")
            self._emit("//")
        self._emit("")
        self._emit_block('''
            #include "unicode.h"

            #include <string>
            #include <vector>
            #include <cstdint>
        ''')

        # Function documentation
        self._emit("/**")
        self._emit(f" * Split text into tokens using the '{func_name}' pattern.")
        self._emit(" *")
        self._emit(" * @param text     UTF-8 encoded input string")
        self._emit(" * @param offsets  Chunk sizes from previous tokenization pass")
        self._emit(" * @return         New chunk sizes after applying this pattern")
        self._emit(" */")
        self._emit(f"std::vector<size_t> unicode_regex_split_{func_name}(")

        with self._indent_block():
            self._emit("const std::string & text,")
            self._emit("const std::vector<size_t> & offsets")

        with self._block(") {"):
            self._emit("std::vector<size_t> bpe_offsets;")
            self._emit("bpe_offsets.reserve(offsets.size());")
            self._emit("")
            self._emit("// Convert UTF-8 to codepoints for pattern matching")
            self._emit("const auto cpts = unicode_cpts_from_utf8(text);")

            # Emit shared backtracking stack if needed
            if self.uses_backtracking:
                self._emit("")
                self._emit("// Pre-allocated backtracking stack for quantifier matching")
                self._emit("// Uses a single vector with base-index tracking to avoid per-match allocations")
                self._emit("std::vector<size_t> bt_stack;")
                self._emit("bt_stack.reserve(cpts.size() * 2);")

            self._emit("")
            self._emit("size_t start = 0;")
            self._emit("")
            self._emit("// Process each chunk from the previous tokenization pass")

            with self._block("for (auto offset : offsets) {"):
                self._emit("const size_t offset_ini = start;")
                self._emit("const size_t offset_end = start + offset;")
                self._emit("start = offset_end;")
                self._emit("")
                self._emit("// Sentinel value for out-of-bounds codepoint access")
                self._emit("static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;")
                self._emit("")
                self._emit("// Helper: Get codepoint at position (returns OUT_OF_RANGE if outside chunk)")

                with self._block("auto _get_cpt = [&](const size_t pos) -> uint32_t {", "};"):
                    self._emit("return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;")
                self._emit("")

                self._emit("// Helper: Get Unicode flags for codepoint at position")
                with self._block("auto _get_flags = [&](const size_t pos) -> unicode_cpt_flags {", "};"):
                    self._emit("return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};")
                self._emit("")

                self._emit("// Helper: Emit a token from _prev_end to 'end'")
                self._emit("size_t _prev_end = offset_ini;")
                with self._block("auto _add_token = [&](const size_t end) -> size_t {", "};"):
                    self._emit_block('''
                        size_t len = end - _prev_end;
                        if (len > 0) {
                            bpe_offsets.push_back(len);
                        }
                        _prev_end = end;
                        return len;
                    ''')
                self._emit("")

                self._emit("// =======================================================")
                self._emit("// Main matching loop")
                self._emit("// Try each alternative in order. First match wins.")
                self._emit("// On match: emit token boundary and continue from new position.")
                self._emit("// On no match: consume single character as fallback.")
                self._emit("// =======================================================")
                with self._block("for (size_t pos = offset_ini; pos < offset_end; ) {"):
                    self._generate_match(ast)
                    self._emit("")
                    self._emit("// No alternative matched - emit single character as token")
                    self._emit("_add_token(++pos);")

            self._emit("")
            self._emit("return bpe_offsets;")

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
        pattern = self._ast_to_pattern(ast)
        self._emit("")
        self._emit(f"// Alternative: {pattern}")

        with self._block():
            self._emit_block('''
                size_t match_pos = pos;
                bool matched = true;
            ''')

            # Generate matching for this alternative
            if isinstance(ast, Sequence):
                self._generate_sequence_match(ast.children)
            else:
                self._generate_node_match(ast)

            self._emit_block('''

                if (matched && match_pos > pos) {
                    pos = match_pos;
                    _add_token(pos);
                    continue;
                }
            ''')

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

    def _generate_literal_match(self, node: LiteralChar, case_insensitive: bool = False):
        char_code = ord(node.char)
        char_desc = self._char_description(node.char)

        if case_insensitive and node.char.isalpha():
            char_lower = ord(node.char.lower())
            self._emit_block('''
                if (matched && unicode_tolower(_get_cpt(match_pos)) == $char_lower) { // $char_desc (case-insensitive)
                    match_pos++;
                } else if (matched) { matched = false; }
            ''', locals())
        else:
            self._emit_block('''
                if (matched && _get_cpt(match_pos) == $char_code) { // $char_desc
                    match_pos++;
                } else if (matched) { matched = false; }
            ''', locals())

    def _generate_special_match(self, node: SpecialChar):
        char_code = ord(node.char)
        char_desc = self._char_description(node.char)
        self._emit_block('''
            if (matched && _get_cpt(match_pos) == $char_code) { // $char_desc
                match_pos++;
            } else if (matched) { matched = false; }
        ''', locals())

    def _generate_charclass_match(self, node: CharClass, case_insensitive: bool = False):
        """Generate match for character class."""
        var = self._temp_var("cpt")
        flags_var = f"flags_{var}"

        # Generate pattern comment for this character class
        class_pattern = self._ast_to_pattern(node)
        ci_suffix = " (case-insensitive)" if case_insensitive else ""
        self._emit(f"// Character class: {class_pattern}{ci_suffix}")

        with self._block("if (matched) {"):
            self._emit(f"uint32_t {var} = _get_cpt(match_pos);")
            self._emit(f"auto {flags_var} = _get_flags(match_pos);")

            # Build conditions with their pattern representations
            conditions = []
            cond_comments = []
            for item in node.items:
                if isinstance(item, tuple):
                    start, end = item
                    if case_insensitive and start.isalpha() and end.isalpha():
                        # Case-insensitive range - compare lowercased
                        conditions.append(f"(unicode_tolower({var}) >= {ord(start.lower())} && unicode_tolower({var}) <= {ord(end.lower())})")
                    else:
                        conditions.append(f"({var} >= {ord(start)} && {var} <= {ord(end)})")
                    cond_comments.append(f"{self._escape_char(start)}-{self._escape_char(end)}")
                elif isinstance(item, LiteralChar):
                    if case_insensitive and item.char.isalpha():
                        # Case-insensitive literal - compare lowercased
                        conditions.append(f"(unicode_tolower({var}) == {ord(item.char.lower())})")
                    else:
                        conditions.append(f"({var} == {ord(item.char)})")
                    cond_comments.append(self._escape_char(item.char))
                elif isinstance(item, SpecialChar):
                    conditions.append(f"({var} == {ord(item.char)})")
                    cond_comments.append(self._escape_char(item.char))
                elif isinstance(item, UnicodeCategory):
                    cond = self._unicode_cat_condition(item, var, flags_var)
                    conditions.append(f"({cond})")
                    p = "P" if item.negated else "p"
                    cond_comments.append(f"\\{p}{{{item.category}}}")
                elif isinstance(item, Predefined):
                    cond = self._predefined_condition(item, var, flags_var)
                    conditions.append(f"({cond})")
                    cond_comments.append(f"\\{item.name}")
                elif isinstance(item, str):
                    if case_insensitive and item.isalpha():
                        conditions.append(f"(unicode_tolower({var}) == {ord(item.lower())})")
                    else:
                        conditions.append(f"({var} == {ord(item)})")
                    cond_comments.append(self._escape_char(item))

            if conditions:
                # Generate multi-line condition with comments
                if node.negated:
                    self._emit(f"// Match if NOT any of: {' | '.join(cond_comments)}")
                    self._emit(f"bool in_class = {conditions[0]}")
                    for cond, comment in zip(conditions[1:], cond_comments[1:]):
                        self._emit(f"    || {cond}")
                    self._emit(";")
                    self._emit_block('''
                        if (!in_class && $var != OUT_OF_RANGE) {
                            match_pos++;
                        } else { matched = false; }
                    ''', locals())
                else:
                    self._emit(f"// Match if any of: {' | '.join(cond_comments)}")
                    if len(conditions) == 1:
                        cond_str = conditions[0]
                        self._emit_block('''
                            if ($cond_str) {
                                match_pos++;
                            } else { matched = false; }
                        ''', locals())
                    else:
                        self._emit(f"if ({conditions[0]}")
                        for cond in conditions[1:]:
                            self._emit(f"    || {cond}")
                        with self._block(") {", "}"):
                            self._emit("match_pos++;")
                        self._emit("else { matched = false; }")
            else:
                if node.negated:
                    self._emit(f"if ({var} != OUT_OF_RANGE) {{ match_pos++; }} else {{ matched = false; }}")
                else:
                    self._emit("matched = false;")

    def _generate_unicode_cat_match(self, node: UnicodeCategory):
        """Generate match for Unicode category."""
        var = self._temp_var("cpt")
        flags_var = f"flags_{var}"
        cond = self._unicode_cat_condition(node, var, flags_var)

        with self._block("if (matched) {"):
            self._emit_block('''
                uint32_t $var = _get_cpt(match_pos);
                auto $flags_var = _get_flags(match_pos);
                if ($cond) {
                    match_pos++;
                } else { matched = false; }
            ''', locals())

    def _unicode_cat_condition(self, node: UnicodeCategory, cpt_var: str, flags_var: str) -> str:
        """Generate condition for Unicode category.

        Unicode General Categories:
        - L  = Letter (Lu | Ll | Lt | Lm | Lo)
        - Lu = Uppercase Letter
        - Ll = Lowercase Letter
        - Lt = Titlecase Letter (rare, e.g., Dž)
        - Lm = Modifier Letter (e.g., ʰ ʱ)
        - Lo = Other Letter (e.g., Chinese, Hebrew, Arabic)
        - M  = Mark (Mn | Mc | Me) - combining marks/accents
        - N  = Number (Nd | Nl | No)
        - P  = Punctuation
        - S  = Symbol
        - Z  = Separator (Zs | Zl | Zp)
        - C  = Other (control, format, etc.)

        Script categories:
        - Han = CJK ideographs
        """
        cat = node.category
        negated = node.negated

        # Map category to flag check
        # Based on unicode_cpt_flags from unicode.h:
        #   is_letter, is_number, is_punctuation, is_symbol, is_accent_mark
        #   is_separator, is_control, is_whitespace, is_uppercase, is_lowercase
        cat_map = {
            # Major categories
            "L": f"{flags_var}.is_letter",
            "N": f"{flags_var}.is_number",
            "P": f"{flags_var}.is_punctuation",
            "S": f"{flags_var}.is_symbol",
            "M": f"{flags_var}.is_accent_mark",
            "Z": f"{flags_var}.is_separator",
            "C": f"{flags_var}.is_control",

            # Letter subcategories
            "Lu": f"({flags_var}.is_letter && {flags_var}.is_uppercase)",
            "Ll": f"({flags_var}.is_letter && {flags_var}.is_lowercase)",
            "Lt": f"({flags_var}.is_letter && {flags_var}.is_uppercase)",  # Titlecase approximated as uppercase
            "Lm": f"({flags_var}.is_letter && !{flags_var}.is_uppercase && !{flags_var}.is_lowercase)",  # Modifier letters
            "Lo": f"({flags_var}.is_letter && !{flags_var}.is_uppercase && !{flags_var}.is_lowercase)",  # Other letters (CJK, etc.)

            # Number subcategories (approximations - all map to is_number)
            "Nd": f"{flags_var}.is_number",  # Decimal digit
            "Nl": f"{flags_var}.is_number",  # Letter number (e.g., Roman numerals)
            "No": f"{flags_var}.is_number",  # Other number

            # Mark subcategories (all map to is_accent_mark)
            "Mn": f"{flags_var}.is_accent_mark",  # Non-spacing mark
            "Mc": f"{flags_var}.is_accent_mark",  # Spacing combining mark
            "Me": f"{flags_var}.is_accent_mark",  # Enclosing mark

            # Script-specific
            "Han": f"unicode_cpt_is_han({cpt_var})",
        }

        self.required_helpers.add(cat)

        if cat in cat_map:
            cond = cat_map[cat]
        else:
            # Unknown category - generate a helper function call
            cond = f"unicode_cpt_is_{cat.lower()}({cpt_var})"
            self.required_helpers.add(f"unicode_cpt_is_{cat.lower()}")

        if negated:
            return f"!({cond})"
        return cond

    def _generate_predefined_match(self, node: Predefined):
        """Generate match for predefined class."""
        var = self._temp_var("cpt")
        flags_var = f"flags_{var}"
        cond = self._predefined_condition(node, var, flags_var)

        with self._block("if (matched) {"):
            self._emit_block('''
                uint32_t $var = _get_cpt(match_pos);
                auto $flags_var = _get_flags(match_pos);
                if ($cond) {
                    match_pos++;
                } else { matched = false; }
            ''', locals())

    def _predefined_condition(self, node: Predefined, cpt_var: str, flags_var: str) -> str:
        """Generate condition for predefined class."""
        name = node.name

        conditions = {
            "s": f"{flags_var}.is_whitespace",
            "S": f"(!{flags_var}.is_whitespace && {flags_var}.as_uint())",
            "d": f"{flags_var}.is_number",
            "D": f"(!{flags_var}.is_number && {flags_var}.as_uint())",
            "w": f"({flags_var}.is_letter || {flags_var}.is_number || {cpt_var} == '_')",
            "W": f"(!({flags_var}.is_letter || {flags_var}.is_number || {cpt_var} == '_') && {flags_var}.as_uint())",
        }

        return conditions.get(name, "false")

    def _generate_any_match(self):
        """Generate match for any character."""
        self._emit_block('''
            if (matched && _get_cpt(match_pos) != OUT_OF_RANGE) {
                match_pos++;
            } else if (matched) { matched = false; }
        ''')

    def _generate_quantifier_match(self, node: Quantifier, case_insensitive: bool = False):
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
            self._emit_block('''
                size_t save_pos = match_pos;
                bool save_matched = matched;
            ''')
            self._generate_node_match(child, case_insensitive)
            self._emit_block('''
                if (!matched) {
                    match_pos = save_pos;
                    matched = save_matched;
                }
            ''')

    def _generate_star_match(self, child: Node, case_insensitive: bool = False):
        """Generate match for zero or more (*)."""
        self._emit("// Zero or more")
        with self._block("while (matched) {"):
            self._emit("size_t save_pos = match_pos;")
            self._generate_node_match(child, case_insensitive)
            self._emit_block('''
                if (!matched || match_pos == save_pos) {
                    match_pos = save_pos;
                    matched = true;
                    break;
                }
            ''')

    def _generate_plus_match(self, child: Node, case_insensitive: bool = False):
        """Generate match for one or more (+)."""
        self._emit("// One or more")
        with self._block():
            self._emit("size_t count = 0;")
            with self._block("while (matched) {"):
                self._emit("size_t save_pos = match_pos;")
                self._generate_node_match(child, case_insensitive)
                self._emit_block('''
                    if (!matched || match_pos == save_pos) {
                        match_pos = save_pos;
                        matched = (count > 0);
                        break;
                    }
                    count++;
                ''')

    def _generate_exact_match(self, child: Node, exact_count: int, case_insensitive: bool = False):
        """Generate match for exact count {n}."""
        self._emit(f"// Exact {exact_count} matches")
        with self._block():
            self._emit("size_t count = 0;")
            with self._block(f"while (matched && count < {exact_count}) {{"):
                self._generate_node_match(child, case_insensitive)
                self._emit("if (matched) count++;")
            self._emit(f"if (count < {exact_count}) matched = false;")

    def _generate_range_match(self, child: Node, min_c: int, max_c: int, case_insensitive: bool = False):
        """Generate match for range {n,m}."""
        if max_c == -1:
            self._emit(f"// {min_c} or more matches")
            with self._block():
                self._emit("size_t count = 0;")
                with self._block("while (matched) {"):
                    self._emit("size_t save_pos = match_pos;")
                    self._generate_node_match(child, case_insensitive)
                    self._emit_block('''
                        if (!matched || match_pos == save_pos) {
                            match_pos = save_pos;
                            matched = (count >= $min_c);
                            break;
                        }
                        count++;
                    ''', locals())
        else:
            self._emit(f"// {min_c} to {max_c} matches")
            with self._block():
                self._emit("size_t count = 0;")
                with self._block(f"while (matched && count < {max_c}) {{"):
                    self._emit("size_t save_pos = match_pos;")
                    self._generate_node_match(child, case_insensitive)
                    self._emit_block('''
                        if (!matched || match_pos == save_pos) {
                            match_pos = save_pos;
                            matched = (count >= $min_c);
                            break;
                        }
                        count++;
                    ''', locals())
                self._emit(f"if (count < {min_c}) matched = false;")

    def _generate_group_match(self, node: GroupNode):
        """Generate match for group."""
        if node.case_insensitive:
            self._emit("// Case-insensitive group")
            self._generate_node_match(node.child, case_insensitive=True)
        else:
            # Just match the child
            self._generate_node_match(node.child)

    def _generate_lookahead_match(self, node: Lookahead, case_insensitive: bool = False):
        """Generate lookahead assertion."""
        lookahead_type = 'Positive' if node.positive else 'Negative'
        self._emit(f"// {lookahead_type} lookahead")

        with self._block():
            self._emit_block('''
                size_t save_match_pos = match_pos;
                bool save_matched = matched;
            ''')

            self._generate_node_match(node.child, case_insensitive)

            self._emit("bool lookahead_success = matched;")
            self._emit("match_pos = save_match_pos;")

            if node.positive:
                self._emit("matched = save_matched && lookahead_success;")
            else:
                self._emit("matched = save_matched && !lookahead_success;")

    def _ast_needs_backtracking(self, ast: Node) -> bool:
        """Pre-scan AST to determine if any part needs backtracking.

        Used to decide whether to emit the shared bt_stack declaration.
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
            return any(self._contains_backtracking_quantifier(a) for a in node.alternatives)
        return False

    def _needs_backtracking(self, children: list) -> bool:
        """Determine if a sequence needs backtracking support.

        Returns True if there's a non-possessive quantifier followed by something that could fail.
        Possessive quantifiers never backtrack, so they don't trigger this.
        """
        has_backtracking_quantifier = False
        for i, child in enumerate(children):
            # Check if this child is a non-possessive quantifier
            is_backtracking_quant = (isinstance(child, Quantifier) and not child.possessive) or \
                       (isinstance(child, GroupNode) and self._contains_backtracking_quantifier(child.child))

            # Check BEFORE updating has_backtracking_quantifier
            if has_backtracking_quantifier:
                # Something after a non-possessive quantifier - might need backtracking
                if isinstance(child, (Lookahead, LiteralChar, CharClass, Predefined,
                                      UnicodeCategory, SpecialChar, AnyChar)):
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

    def _generate_sequence_with_backtracking(self, children: list, case_insensitive: bool = False):
        """Generate sequence matching with stack-based backtracking.

        Strategy: Use shared bt_stack with base index tracking.
        For each non-possessive quantifier, collect ALL possible match positions into bt_stack.
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

        pattern_str = "".join(self._ast_to_pattern(c) for c in children[:5])
        if len(children) > 5:
            pattern_str += "..."

        num_quants = len(quantifier_indices)

        self._emit(f"// Sequence with backtracking ({num_quants} quantifiers): {pattern_str}")
        with self._block():
            self._emit("bool seq_matched = false;")
            self._emit("size_t bt_base = bt_stack.size();  // Save stack state")
            self._emit("")

            # Generate elements before first quantifier
            first_quant_idx = quantifier_indices[0]
            for i in range(first_quant_idx):
                self._generate_node_match(children[i], case_insensitive)
                self._emit("if (!matched) { bt_stack.resize(bt_base); }")
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
            self._emit("bt_stack.resize(bt_base);  // Restore stack state")
            self._emit("matched = seq_matched;")

    def _generate_stack_based_backtracking(self, children: list, quant_indices: list,
                                            quant_num: int, case_insensitive: bool):
        """Generate stack-based nested loops for multiple quantifiers.

        Uses shared bt_stack with base index tracking instead of per-quantifier vectors.
        For each quantifier, we:
        1. Collect all possible match positions into bt_stack
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
        quant_pattern = self._ast_to_pattern(quant)

        # Collect positions for this quantifier using shared stack
        self._emit(f"// Quantifier {quant_num}: {quant_pattern}")
        self._emit(f"size_t {base_name} = bt_stack.size();")
        self._emit("bt_stack.push_back(match_pos);")

        with self._block():
            if max_c == -1:
                loop_cond = "while (true) {"
            else:
                loop_cond = f"while (bt_stack.size() - {base_name} <= {max_c}) {{"
            with self._block(loop_cond):
                self._emit_block('''
                    size_t save_pos = match_pos;
                    matched = true;
                ''')
                self._generate_node_match(quant.child, case_insensitive)
                self._emit_block('''
                    if (matched && match_pos > save_pos) {
                        bt_stack.push_back(match_pos);
                    } else {
                        match_pos = save_pos;
                        break;
                    }
                ''')
        self._emit(f"size_t {count_name} = bt_stack.size() - {base_name};")
        self._emit("")

        # Loop through positions - direction depends on greedy vs lazy
        if greedy:
            # Greedy: try longest match first, backtrack to shorter
            self._emit(f"// Try quantifier {quant_num} positions longest-first (greedy, min_count={min_c})")
            loop_header = f"for (size_t i{quant_num} = {count_name}; i{quant_num} > {min_c}; i{quant_num}--) {{"
        else:
            # Lazy: try shortest match first, extend if needed
            self._emit(f"// Try quantifier {quant_num} positions shortest-first (lazy, min_count={min_c})")
            loop_header = f"for (size_t i{quant_num} = {min_c} + 1; i{quant_num} <= {count_name}; i{quant_num}++) {{"
        with self._block(loop_header):
            self._emit(f"match_pos = bt_stack[{base_name} + i{quant_num} - 1];")
            self._emit("matched = true;")

            # Find elements between this quantifier and the next (or end)
            next_quant_idx = quant_indices[quant_num + 1] if quant_num + 1 < len(quant_indices) else len(children)

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
                self._emit(f"bt_stack.resize({next_base});")
                self._emit("if (seq_matched) break;")
            else:
                for i in range(next_quant_idx, len(children)):
                    self._generate_node_match(children[i], case_insensitive)
                    if i < len(children) - 1:
                        self._emit("if (!matched) continue;")
                        self._emit("")
                self._emit("if (matched) { seq_matched = true; break; }")

    def _generate_nested_alternation(self, node: Alternation, case_insensitive: bool = False):
        """Generate nested alternation (within a sequence)."""
        self._emit("// Nested alternation")
        with self._block("if (matched) {"):
            self._emit_block('''
                size_t alt_save = match_pos;
                bool alt_matched = false;
            ''')

            for i, alt in enumerate(node.alternatives):
                if i > 0:
                    self._emit("if (!alt_matched) {")
                    self.indent_level += 1
                    self._emit_block('''
                        match_pos = alt_save;
                        matched = true;
                    ''')

                if isinstance(alt, Sequence):
                    for child in alt.children:
                        self._generate_node_match(child, case_insensitive)
                else:
                    self._generate_node_match(alt, case_insensitive)

                self._emit("if (matched) alt_matched = true;")

                if i > 0:
                    self.indent_level -= 1
                    self._emit("}")

            self._emit("matched = alt_matched;")

    def _escape_char(self, c: str) -> str:
        """Escape character for C++ comment."""
        if c == '\r':
            return '\\r'
        elif c == '\n':
            return '\\n'
        elif c == '\t':
            return '\\t'
        elif c == '\\':
            return '\\\\'
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

    def _charclass_items_to_pattern(self, items: list) -> str:
        """Convert character class items to pattern string."""
        parts = []
        for item in items:
            if isinstance(item, tuple):
                # Range like ('a', 'z')
                start, end = item
                parts.append(f"{self._escape_char(start)}-{self._escape_char(end)}")
            elif isinstance(item, LiteralChar):
                c = item.char
                if c in "\\[]^-":
                    parts.append(f"\\{c}")
                else:
                    parts.append(self._escape_char(c))
            elif isinstance(item, SpecialChar):
                parts.append(self._escape_char(item.char))
            elif isinstance(item, UnicodeCategory):
                p = "P" if item.negated else "p"
                parts.append(f"\\{p}{{{item.category}}}")
            elif isinstance(item, Predefined):
                parts.append(f"\\{item.name}")
            elif isinstance(item, str):
                if item in "\\[]^-":
                    parts.append(f"\\{item}")
                else:
                    parts.append(self._escape_char(item))
        return "".join(parts)

    def _ast_to_pattern(self, ast: Node, depth: int = 0) -> str:
        """Convert AST back to a pattern string for comments."""
        if depth > 5:
            return "..."

        if isinstance(ast, LiteralChar):
            if ast.char in "\\[]{}()*+?|^$.":
                return f"\\{ast.char}"
            return ast.char
        elif isinstance(ast, CharClass):
            neg = "^" if ast.negated else ""
            items_str = self._charclass_items_to_pattern(ast.items)
            return f"[{neg}{items_str}]"
        elif isinstance(ast, UnicodeCategory):
            p = "P" if ast.negated else "p"
            return f"\\{p}{{{ast.category}}}"
        elif isinstance(ast, Predefined):
            return f"\\{ast.name}"
        elif isinstance(ast, SpecialChar):
            return {'\r': '\\r', '\n': '\\n', '\t': '\\t'}.get(ast.char, repr(ast.char))
        elif isinstance(ast, AnyChar):
            return "."
        elif isinstance(ast, Quantifier):
            child = self._ast_to_pattern(ast.child, depth + 1)
            # Build base quantifier
            if ast.min_count == 0 and ast.max_count == 1:
                q = "?"
            elif ast.min_count == 0 and ast.max_count == -1:
                q = "*"
            elif ast.min_count == 1 and ast.max_count == -1:
                q = "+"
            elif ast.min_count == ast.max_count:
                q = f"{{{ast.min_count}}}"
            elif ast.max_count == -1:
                q = f"{{{ast.min_count},}}"
            else:
                q = f"{{{ast.min_count},{ast.max_count}}}"
            # Add lazy or possessive suffix
            if ast.possessive:
                q += "+"
            elif not ast.greedy:
                q += "?"
            return f"{child}{q}"
        elif isinstance(ast, Alternation):
            alts = [self._ast_to_pattern(a, depth + 1) for a in ast.alternatives]
            alt_str = "|".join(alts)
            # Wrap in parentheses if nested (not at top level)
            if depth > 0:
                return f"({alt_str})"
            return alt_str
        elif isinstance(ast, Sequence):
            parts = [self._ast_to_pattern(c, depth + 1) for c in ast.children]
            return "".join(parts)
        elif isinstance(ast, GroupNode):
            child = self._ast_to_pattern(ast.child, depth + 1)
            if ast.case_insensitive:
                return f"(?i:{child})"
            elif not ast.capturing:
                return f"(?:{child})"
            else:
                return f"({child})"
        elif isinstance(ast, Lookahead):
            child = self._ast_to_pattern(ast.child, depth + 1)
            op = "=" if ast.positive else "!"
            return f"(?{op}{child})"
        elif isinstance(ast, Anchor):
            return "^" if ast.type == "start" else "$"
        else:
            return str(ast)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert PCRE patterns to C++ code"
    )
    parser.add_argument(
        "--pattern", "-p",
        required=True,
        help="The PCRE pattern to convert"
    )
    parser.add_argument(
        "--name", "-n",
        required=True,
        help="Name for the generated function (e.g., 'gpt2', 'llama3')"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)"
    )

    args = parser.parse_args()

    # Parse the pattern
    try:
        ast = parse_pcre(args.pattern)
    except ValueError as e:
        print(f"Error parsing pattern: {e}", file=sys.stderr)
        sys.exit(1)

    # Optimize the AST
    ast = ASTOptimizer().optimize(ast)

    # Generate C++ code
    emitter = CppEmitter(args.name)
    cpp_code = emitter.generate(ast, pattern=args.pattern)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(cpp_code)
        print(f"Generated C++ code written to {args.output}")
    else:
        print(cpp_code)

    # Report required helpers
    if emitter.required_helpers:
        print("\n// Required helper functions:", file=sys.stderr)
        for helper in sorted(emitter.required_helpers):
            print(f"//   - {helper}", file=sys.stderr)


if __name__ == "__main__":
    main()
