#!/usr/bin/env python3
"""
PCRE to C++ Converter

Converts PCRE regular expressions into C++ functions that split input text
into chunks for LLM pretokenization.
"""

import argparse
import re
import sys
from dataclasses import dataclass
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
            min_c, max_c, greedy = quantifier
            return Quantifier(atom, min_c, max_c, greedy)

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

    def _parse_quantifier(self) -> Optional[Tuple[int, int, bool]]:
        """Parse quantifier: *, +, ?, {n}, {n,}, {n,m}"""
        ch = self._peek()

        if ch == '*':
            self._advance()
            greedy = self._parse_lazy_modifier()
            return (0, -1, greedy)

        if ch == '+':
            self._advance()
            greedy = self._parse_lazy_modifier()
            return (1, -1, greedy)

        if ch == '?':
            self._advance()
            greedy = self._parse_lazy_modifier()
            return (0, 1, greedy)

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
                greedy = self._parse_lazy_modifier()
                return (min_val, min_val, greedy)

            if self._peek() == ',':
                self._advance()

                if self._peek() == '}':
                    # {n,}
                    self._advance()
                    greedy = self._parse_lazy_modifier()
                    return (min_val, -1, greedy)

                # {n,m}
                max_start = self.pos
                while self._peek() and self._peek().isdigit():
                    self._advance()
                max_val = int(self.pattern[max_start:self.pos])

                self._expect('}')
                greedy = self._parse_lazy_modifier()
                return (min_val, max_val, greedy)

            raise ValueError(f"Invalid quantifier at position {self.pos}")

        return None

    def _parse_lazy_modifier(self) -> bool:
        """Parse optional lazy modifier '?'. Returns True if greedy."""
        if self._peek() == '?':
            self._advance()
            return False  # lazy
        return True  # greedy


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

    def _indent(self):
        return "    " * self.indent_level

    def _emit(self, line: str = ""):
        self.lines.append(self._indent() + line)

    def _temp_var(self, prefix: str = "tmp") -> str:
        self.temp_counter += 1
        return f"{prefix}_{self.temp_counter}"

    def generate(self, ast: Node) -> str:
        """Generate C++ code for the given AST."""
        self.lines = []
        self.required_helpers = set()

        # File header
        self._emit("// Auto-generated by pcre_to_cpp.py")
        self._emit("// Do not edit manually")
        self._emit("")
        self._emit("#include \"unicode.h\"")
        self._emit("")
        self._emit("#include <string>")
        self._emit("#include <vector>")
        self._emit("#include <cstdint>")
        self._emit("")

        # Function signature
        self._emit(f"std::vector<size_t> unicode_regex_split_{self.function_name}(")
        self.indent_level += 1
        self._emit("const std::string & text,")
        self._emit("const std::vector<size_t> & offsets")
        self.indent_level -= 1
        self._emit(") {")
        self.indent_level += 1

        # Function body setup
        self._emit("std::vector<size_t> bpe_offsets;")
        self._emit("bpe_offsets.reserve(offsets.size());")
        self._emit("")
        self._emit("const auto cpts = unicode_cpts_from_utf8(text);")
        self._emit("")
        self._emit("size_t start = 0;")
        self._emit("for (auto offset : offsets) {")
        self.indent_level += 1

        self._emit("const size_t offset_ini = start;")
        self._emit("const size_t offset_end = start + offset;")
        self._emit("start = offset_end;")
        self._emit("")

        # Helper lambdas
        self._emit("static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;")
        self._emit("auto _get_cpt = [&](const size_t pos) -> uint32_t {")
        self.indent_level += 1
        self._emit("return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;")
        self.indent_level -= 1
        self._emit("};")
        self._emit("")

        self._emit("auto _get_flags = [&](const size_t pos) -> unicode_cpt_flags {")
        self.indent_level += 1
        self._emit("return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags_from_cpt(cpts[pos]) : unicode_cpt_flags{};")
        self.indent_level -= 1
        self._emit("};")
        self._emit("")

        self._emit("size_t _prev_end = offset_ini;")
        self._emit("auto _add_token = [&](const size_t end) -> size_t {")
        self.indent_level += 1
        self._emit("size_t len = end - _prev_end;")
        self._emit("if (len > 0) {")
        self.indent_level += 1
        self._emit("bpe_offsets.push_back(len);")
        self.indent_level -= 1
        self._emit("}")
        self._emit("_prev_end = end;")
        self._emit("return len;")
        self.indent_level -= 1
        self._emit("};")
        self._emit("")

        # Main matching loop
        self._emit("for (size_t pos = offset_ini; pos < offset_end; ) {")
        self.indent_level += 1

        # Generate matching code
        self._generate_match(ast)

        # Fallback: no match, consume single character
        self._emit("")
        self._emit("// No match - consume single character")
        self._emit("_add_token(++pos);")

        self.indent_level -= 1
        self._emit("}")  # end for pos

        self.indent_level -= 1
        self._emit("}")  # end for offset
        self._emit("")
        self._emit("return bpe_offsets;")

        self.indent_level -= 1
        self._emit("}")  # end function

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
        self._emit(f"// Alternative: {self._ast_to_pattern(ast)}")
        self._emit("{")
        self.indent_level += 1

        self._emit("size_t match_pos = pos;")
        self._emit("bool matched = true;")
        self._emit("")

        # Generate matching for this alternative
        if isinstance(ast, Sequence):
            for child in ast.children:
                self._generate_node_match(child)
        else:
            self._generate_node_match(ast)

        self._emit("")
        self._emit("if (matched && match_pos > pos) {")
        self.indent_level += 1
        self._emit("pos = match_pos;")
        self._emit("_add_token(pos);")
        self._emit("continue;")
        self.indent_level -= 1
        self._emit("}")

        self.indent_level -= 1
        self._emit("}")

    def _generate_node_match(self, node: Node, case_insensitive: bool = False):
        """Generate matching code for a single node."""
        if isinstance(node, LiteralChar):
            self._generate_literal_match(node, case_insensitive)
        elif isinstance(node, CharClass):
            self._generate_charclass_match(node)
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
            for child in node.children:
                self._generate_node_match(child, case_insensitive)
        elif isinstance(node, GroupNode):
            self._generate_group_match(node)
        elif isinstance(node, Lookahead):
            self._generate_lookahead_match(node)
        elif isinstance(node, Alternation):
            self._generate_nested_alternation(node, case_insensitive)
        elif isinstance(node, Anchor):
            # Anchors are typically no-ops in this context
            pass
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    def _generate_literal_match(self, node: LiteralChar, case_insensitive: bool = False):
        char_code = ord(node.char)
        if case_insensitive and node.char.isalpha():
            char_lower = ord(node.char.lower())
            self._emit(f"if (matched && unicode_tolower(_get_cpt(match_pos)) == {char_lower}) {{ // '{self._escape_char(node.char)}' (case-insensitive)")
            self.indent_level += 1
            self._emit("match_pos++;")
            self.indent_level -= 1
            self._emit("} else if (matched) { matched = false; }")
        else:
            self._emit(f"if (matched && _get_cpt(match_pos) == {char_code}) {{ // '{self._escape_char(node.char)}'")
            self.indent_level += 1
            self._emit("match_pos++;")
            self.indent_level -= 1
            self._emit("} else if (matched) { matched = false; }")

    def _generate_special_match(self, node: SpecialChar):
        char_code = ord(node.char)
        char_name = {'\r': '\\r', '\n': '\\n', '\t': '\\t'}.get(node.char, repr(node.char))
        self._emit(f"if (matched && _get_cpt(match_pos) == {char_code}) {{ // {char_name}")
        self.indent_level += 1
        self._emit("match_pos++;")
        self.indent_level -= 1
        self._emit("} else if (matched) { matched = false; }")

    def _generate_charclass_match(self, node: CharClass):
        """Generate match for character class."""
        var = self._temp_var("cpt")
        self._emit(f"if (matched) {{")
        self.indent_level += 1
        self._emit(f"uint32_t {var} = _get_cpt(match_pos);")
        self._emit(f"auto flags_{var} = _get_flags(match_pos);")

        conditions = []
        for item in node.items:
            if isinstance(item, tuple):
                # Range
                start, end = item
                conditions.append(f"({var} >= {ord(start)} && {var} <= {ord(end)})")
            elif isinstance(item, LiteralChar):
                conditions.append(f"({var} == {ord(item.char)})")
            elif isinstance(item, SpecialChar):
                conditions.append(f"({var} == {ord(item.char)})")
            elif isinstance(item, UnicodeCategory):
                cond = self._unicode_cat_condition(item, var, f"flags_{var}")
                conditions.append(f"({cond})")
            elif isinstance(item, Predefined):
                cond = self._predefined_condition(item, var, f"flags_{var}")
                conditions.append(f"({cond})")
            elif isinstance(item, str):
                # Single character as string
                conditions.append(f"({var} == {ord(item)})")

        if conditions:
            cond_str = " || ".join(conditions)
            if node.negated:
                self._emit(f"bool in_class = {cond_str};")
                self._emit(f"if (!in_class && {var} != OUT_OF_RANGE) {{")
            else:
                self._emit(f"if ({cond_str}) {{")
            self.indent_level += 1
            self._emit("match_pos++;")
            self.indent_level -= 1
            self._emit("} else { matched = false; }")
        else:
            if node.negated:
                self._emit(f"if ({var} != OUT_OF_RANGE) {{ match_pos++; }} else {{ matched = false; }}")
            else:
                self._emit("matched = false;")

        self.indent_level -= 1
        self._emit("}")

    def _generate_unicode_cat_match(self, node: UnicodeCategory):
        """Generate match for Unicode category."""
        var = self._temp_var("cpt")
        self._emit(f"if (matched) {{")
        self.indent_level += 1
        self._emit(f"uint32_t {var} = _get_cpt(match_pos);")
        self._emit(f"auto flags_{var} = _get_flags(match_pos);")

        cond = self._unicode_cat_condition(node, var, f"flags_{var}")

        self._emit(f"if ({cond}) {{")
        self.indent_level += 1
        self._emit("match_pos++;")
        self.indent_level -= 1
        self._emit("} else { matched = false; }")

        self.indent_level -= 1
        self._emit("}")

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
        self._emit(f"if (matched) {{")
        self.indent_level += 1
        self._emit(f"uint32_t {var} = _get_cpt(match_pos);")
        self._emit(f"auto flags_{var} = _get_flags(match_pos);")

        cond = self._predefined_condition(node, var, f"flags_{var}")

        self._emit(f"if ({cond}) {{")
        self.indent_level += 1
        self._emit("match_pos++;")
        self.indent_level -= 1
        self._emit("} else { matched = false; }")

        self.indent_level -= 1
        self._emit("}")

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
        self._emit("if (matched && _get_cpt(match_pos) != OUT_OF_RANGE) {")
        self.indent_level += 1
        self._emit("match_pos++;")
        self.indent_level -= 1
        self._emit("} else if (matched) { matched = false; }")

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
        self._emit("{")
        self.indent_level += 1
        self._emit("size_t save_pos = match_pos;")
        self._emit("bool save_matched = matched;")
        self._generate_node_match(child, case_insensitive)
        self._emit("if (!matched) {")
        self.indent_level += 1
        self._emit("match_pos = save_pos;")
        self._emit("matched = save_matched;")
        self.indent_level -= 1
        self._emit("}")
        self.indent_level -= 1
        self._emit("}")

    def _generate_star_match(self, child: Node, case_insensitive: bool = False):
        """Generate match for zero or more (*)."""
        self._emit("// Zero or more")
        self._emit("while (matched) {")
        self.indent_level += 1
        self._emit("size_t save_pos = match_pos;")
        self._generate_node_match(child, case_insensitive)
        self._emit("if (!matched || match_pos == save_pos) {")
        self.indent_level += 1
        self._emit("match_pos = save_pos;")
        self._emit("matched = true;")
        self._emit("break;")
        self.indent_level -= 1
        self._emit("}")
        self.indent_level -= 1
        self._emit("}")

    def _generate_plus_match(self, child: Node, case_insensitive: bool = False):
        """Generate match for one or more (+)."""
        self._emit("// One or more")
        self._emit("{")
        self.indent_level += 1
        self._emit("size_t count = 0;")
        self._emit("while (matched) {")
        self.indent_level += 1
        self._emit("size_t save_pos = match_pos;")
        self._generate_node_match(child, case_insensitive)
        self._emit("if (!matched || match_pos == save_pos) {")
        self.indent_level += 1
        self._emit("match_pos = save_pos;")
        self._emit("matched = (count > 0);")
        self._emit("break;")
        self.indent_level -= 1
        self._emit("}")
        self._emit("count++;")
        self.indent_level -= 1
        self._emit("}")
        self.indent_level -= 1
        self._emit("}")

    def _generate_exact_match(self, child: Node, count: int, case_insensitive: bool = False):
        """Generate match for exact count {n}."""
        self._emit(f"// Exact {count} matches")
        self._emit("{")
        self.indent_level += 1
        self._emit(f"size_t count = 0;")
        self._emit(f"while (matched && count < {count}) {{")
        self.indent_level += 1
        self._generate_node_match(child, case_insensitive)
        self._emit("if (matched) count++;")
        self.indent_level -= 1
        self._emit("}")
        self._emit(f"if (count < {count}) matched = false;")
        self.indent_level -= 1
        self._emit("}")

    def _generate_range_match(self, child: Node, min_c: int, max_c: int, case_insensitive: bool = False):
        """Generate match for range {n,m}."""
        if max_c == -1:
            self._emit(f"// {min_c} or more matches")
            self._emit("{")
            self.indent_level += 1
            self._emit("size_t count = 0;")
            self._emit("while (matched) {")
            self.indent_level += 1
            self._emit("size_t save_pos = match_pos;")
            self._generate_node_match(child, case_insensitive)
            self._emit("if (!matched || match_pos == save_pos) {")
            self.indent_level += 1
            self._emit("match_pos = save_pos;")
            self._emit(f"matched = (count >= {min_c});")
            self._emit("break;")
            self.indent_level -= 1
            self._emit("}")
            self._emit("count++;")
            self.indent_level -= 1
            self._emit("}")
            self.indent_level -= 1
            self._emit("}")
        else:
            self._emit(f"// {min_c} to {max_c} matches")
            self._emit("{")
            self.indent_level += 1
            self._emit("size_t count = 0;")
            self._emit(f"while (matched && count < {max_c}) {{")
            self.indent_level += 1
            self._emit("size_t save_pos = match_pos;")
            self._generate_node_match(child, case_insensitive)
            self._emit("if (!matched || match_pos == save_pos) {")
            self.indent_level += 1
            self._emit("match_pos = save_pos;")
            self._emit(f"matched = (count >= {min_c});")
            self._emit("break;")
            self.indent_level -= 1
            self._emit("}")
            self._emit("count++;")
            self.indent_level -= 1
            self._emit("}")
            self._emit(f"if (count < {min_c}) matched = false;")
            self.indent_level -= 1
            self._emit("}")

    def _generate_group_match(self, node: GroupNode):
        """Generate match for group."""
        if node.case_insensitive:
            self._emit("// Case-insensitive group")
            self._generate_node_match(node.child, case_insensitive=True)
        else:
            # Just match the child
            self._generate_node_match(node.child)

    def _generate_lookahead_match(self, node: Lookahead):
        """Generate lookahead assertion."""
        self._emit(f"// {'Positive' if node.positive else 'Negative'} lookahead")
        self._emit("{")
        self.indent_level += 1
        self._emit("size_t save_match_pos = match_pos;")
        self._emit("bool save_matched = matched;")

        # Match the lookahead pattern
        self._generate_node_match(node.child)

        self._emit("bool lookahead_success = matched;")

        # Restore position (lookahead doesn't consume)
        self._emit("match_pos = save_match_pos;")

        if node.positive:
            self._emit("matched = save_matched && lookahead_success;")
        else:
            self._emit("matched = save_matched && !lookahead_success;")

        self.indent_level -= 1
        self._emit("}")

    def _generate_nested_alternation(self, node: Alternation, case_insensitive: bool = False):
        """Generate nested alternation (within a sequence)."""
        self._emit("// Nested alternation")
        self._emit("{")
        self.indent_level += 1
        self._emit("size_t alt_save = match_pos;")
        self._emit("bool alt_matched = false;")

        for i, alt in enumerate(node.alternatives):
            if i > 0:
                self._emit("if (!alt_matched) {")
                self.indent_level += 1
                self._emit("match_pos = alt_save;")
                self._emit("matched = true;")

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
        self.indent_level -= 1
        self._emit("}")

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
            return f"[{neg}...]"
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
            if ast.min_count == 0 and ast.max_count == 1:
                return f"{child}?"
            elif ast.min_count == 0 and ast.max_count == -1:
                return f"{child}*"
            elif ast.min_count == 1 and ast.max_count == -1:
                return f"{child}+"
            elif ast.min_count == ast.max_count:
                return f"{child}{{{ast.min_count}}}"
            elif ast.max_count == -1:
                return f"{child}{{{ast.min_count},}}"
            else:
                return f"{child}{{{ast.min_count},{ast.max_count}}}"
        elif isinstance(ast, Alternation):
            alts = [self._ast_to_pattern(a, depth + 1) for a in ast.alternatives]
            return "|".join(alts)
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

    # Generate C++ code
    emitter = CppEmitter(args.name)
    cpp_code = emitter.generate(ast)

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
