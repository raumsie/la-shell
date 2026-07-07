"""Lexer for the Linear Algebra Shell.

Tokenizes an input string (potentially spanning multiple lines) into a flat
list of Token objects using a single-pass regex scanner.  The token stream
always terminates with an EOF token.

Supported token types:
    NUMBER      – integer or floating-point literal (stored as Python float)
    IDENTIFIER  – letter/underscore-started name that is NOT a function keyword
    LBRACKET    – '['
    RBRACKET    – ']'
    SEMICOLON   – ';'  (row separator inside matrix literals)
    COMMA       – ','  (column separator inside matrix literals / function args)
    EQUALS      – '='  (assignment)
    PLUS        – '+'
    MINUS       – '-'
    STAR        – '*'  (matrix multiplication in the evaluator)
    LPAREN      – '('
    RPAREN      – ')'
    FUNCTION    – one of: transpose det inv eye zeros ones trace
                  dagger outer tensor kron commutator
    DOT         – '.'  (prefix for shell commands such as .help)
    EOF         – sentinel appended at end of input

Design notes:
    - A single compiled regex is used for efficiency (one pass over the input).
    - Named capture groups map directly to TokenType names.
    - Whitespace (spaces, tabs, carriage returns) and newlines are discarded;
      newlines still increment the line counter for error reporting.
    - The WORD group captures any letter-started identifier; the Lexer then
      checks whether the matched value is a reserved function keyword before
      deciding which TokenType to emit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Token type enumeration
# ---------------------------------------------------------------------------

class TokenType(Enum):
    """All token categories recognized by the LA Shell lexer.

    Each member maps to a distinct syntactic role.  The integer values are
    assigned automatically and should not be relied upon by callers.
    """

    NUMBER = auto()      # 3.14, 42, 0.5, .75
    IDENTIFIER = auto()  # A, B, my_var, x1
    LBRACKET = auto()    # [
    RBRACKET = auto()    # ]
    SEMICOLON = auto()   # ;
    COMMA = auto()       # ,
    EQUALS = auto()      # =
    PLUS = auto()        # +
    MINUS = auto()       # -
    STAR = auto()        # *
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    FUNCTION = auto()    # transpose | det | inv | eye | zeros | ones | trace
                         # | dagger | outer | tensor | kron | commutator
    DOT = auto()         # .
    EOF = auto()         # end-of-input sentinel


# ---------------------------------------------------------------------------
# Reserved function keywords
# ---------------------------------------------------------------------------

FUNCTION_KEYWORDS: frozenset[str] = frozenset({
    'transpose',
    'det',
    'inv',
    'eye',
    'zeros',
    'ones',
    'trace',
    'dagger',
    'outer',
    'tensor',
    'kron',
    'commutator',
})

# ---------------------------------------------------------------------------
# Master regex pattern
# ---------------------------------------------------------------------------
# Order matters: alternatives are tried left-to-right, so more specific
# patterns (e.g., floats before integers) must appear first.
# Named groups correspond to logical token categories; the WORD group covers
# both identifiers and function keywords – the Lexer resolves the distinction.

_TOKEN_PATTERN = re.compile(
    r'(?P<NUMBER>\d+\.\d*|\.\d+|\d+)'       # float: 1.5  .5  1.  or int: 42
    r'|(?P<WORD>[a-zA-Z_][a-zA-Z0-9_]*)'    # identifier / keyword
    r'|(?P<LBRACKET>\[)'
    r'|(?P<RBRACKET>\])'
    r'|(?P<SEMICOLON>;)'
    r'|(?P<COMMA>,)'
    r'|(?P<EQUALS>=)'
    r'|(?P<PLUS>\+)'
    r'|(?P<MINUS>-)'
    r'|(?P<STAR>\*)'
    r'|(?P<LPAREN>\()'
    r'|(?P<RPAREN>\))'
    r'|(?P<DOT>\.)'
    r'|(?P<NEWLINE>\n)'
    r'|(?P<WHITESPACE>[ \t\r]+)'
    r'|(?P<MISMATCH>.)',                     # catch-all for illegal characters
    re.DOTALL,
)

# Mapping from named-group identifiers that translate 1-to-1 to TokenType names
_DIRECT_TOKEN_KINDS: frozenset[str] = frozenset(
    t.name for t in TokenType
    if t not in (TokenType.NUMBER, TokenType.IDENTIFIER, TokenType.FUNCTION, TokenType.EOF)
)


# ---------------------------------------------------------------------------
# Token dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    """An immutable lexical token.

    Attributes:
        type:     The syntactic category of this token.
        value:    The matched text for most token types; a Python ``float``
                  for NUMBER tokens.
        position: Zero-based character offset of the first character of this
                  token within the source string.
        line:     One-based line number where the token begins.
    """

    type: TokenType
    value: str | float
    position: int
    line: int

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Token({self.type.name}, {self.value!r}, "
            f"line={self.line}, pos={self.position})"
        )


# ---------------------------------------------------------------------------
# Lexer error
# ---------------------------------------------------------------------------

class LexerError(Exception):
    """Raised when the Lexer encounters a character it cannot tokenize.

    Attributes:
        message:  Human-readable description.
        position: Zero-based offset in the source string.
        line:     One-based line number.
    """

    def __init__(self, message: str, position: int, line: int) -> None:
        super().__init__(message)
        self.position = position
        self.line = line


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

class Lexer:
    """Regex-based tokenizer for the Linear Algebra Shell.

    Scans the entire source string in a single pass using a compiled regex
    with named groups.  Whitespace (including newlines) is discarded but
    newlines increment the internal line counter for accurate error reporting.

    Identifiers that match one of the reserved function keywords
    (``transpose``, ``det``, ``inv``, ``eye``, ``zeros``, ``ones``,
    ``trace``, ``dagger``, ``outer``, ``tensor``, ``kron``, ``commutator``)
    are emitted as FUNCTION tokens rather than IDENTIFIER tokens.

    A single EOF token is always appended at the end of the token list.

    Example::

        >>> tokens = Lexer('[1.0, 2; 3, 4]').tokenize()
        >>> [t.type.name for t in tokens]
        ['LBRACKET', 'NUMBER', 'COMMA', 'NUMBER', 'SEMICOLON',
         'NUMBER', 'COMMA', 'NUMBER', 'RBRACKET', 'EOF']

        >>> tokens = Lexer('transpose(A)').tokenize()
        >>> [(t.type.name, t.value) for t in tokens]
        [('FUNCTION', 'transpose'), ('LPAREN', '('), ('IDENTIFIER', 'A'),
         ('RPAREN', ')'), ('EOF', '')]
    """

    def __init__(self, source: str) -> None:
        """Initialise the Lexer.

        Args:
            source: Raw input text to tokenize.  May span multiple lines.
        """
        self.source: str = source

    def tokenize(self) -> list[Token]:
        """Scan the full source string and return the complete token list.

        Returns:
            A list of Token objects.  The final element is always an EOF token
            whose ``value`` is the empty string.

        Raises:
            LexerError: When an unrecognised character is found in the source.
        """
        tokens: list[Token] = []
        line: int = 1

        for match in _TOKEN_PATTERN.finditer(self.source):
            kind: str = match.lastgroup  # type: ignore[assignment]
            raw: str = match.group()
            pos: int = match.start()

            # ---- skip silently ----
            if kind == 'NEWLINE':
                line += 1
                continue
            if kind == 'WHITESPACE':
                continue

            # ---- illegal character ----
            if kind == 'MISMATCH':
                raise LexerError(
                    f"Unexpected character {raw!r} at position {pos} (line {line})",
                    position=pos,
                    line=line,
                )

            # ---- numeric literal ----
            if kind == 'NUMBER':
                tokens.append(Token(TokenType.NUMBER, float(raw), pos, line))
                continue

            # ---- identifier or function keyword ----
            if kind == 'WORD':
                token_type = (
                    TokenType.FUNCTION if raw in FUNCTION_KEYWORDS
                    else TokenType.IDENTIFIER
                )
                tokens.append(Token(token_type, raw, pos, line))
                continue

            # ---- single-character tokens with 1-to-1 group-name mapping ----
            # The group name (e.g. 'LBRACKET') is the same as the TokenType
            # member name, so we can look it up directly.
            tokens.append(Token(TokenType[kind], raw, pos, line))

        # Always close the stream with EOF
        eof_pos: int = len(self.source)
        tokens.append(Token(TokenType.EOF, '', eof_pos, line))
        return tokens
