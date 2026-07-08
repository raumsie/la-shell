"""Recursive descent parser for the Linear Algebra Shell.

Consumes a flat list of Token objects produced by ``Lexer.tokenize()`` and
builds an Abstract Syntax Tree (AST) composed entirely of ``dataclass`` nodes.
The parser raises ``SyntaxError`` for any grammatically invalid input, giving
positional information so the REPL can display a helpful message.

Formal grammar (EBNF)
---------------------
    program     ::= assignment | expression
    assignment  ::= IDENTIFIER '=' expression
    expression  ::= term (('+' | '-') term)*
    term        ::= factor ('*' factor)*
    factor      ::= NUMBER
                  | FUNCTION '(' expression (',' expression)* ')'
                  | IDENTIFIER
                  | '[' row (';' row)* ']'
                  | '-' factor
                  | '(' expression ')'
    row         ::= expression (',' expression)*

Operator precedence (lowest → highest)
---------------------------------------
    +  -    additive,       left-associative   (parse_expression)
    *       multiplicative, left-associative   (parse_term)
    unary - prefix negation, right-associative (parse_factor)

AST node classes
----------------
All nodes are frozen ``dataclass`` instances, making them hashable and
preventing accidental mutation by the evaluator.

    NumberNode      – numeric literal (float)
    VarNode         – variable reference (str)
    MatrixNode      – matrix literal [[row0], [row1], ...]
    BinOpNode       – binary operation (left op right)
    UnaryNode       – unary operation (-operand)
    FunctionNode    – built-in function call (name, args)
    AssignmentNode  – variable assignment (name = expr)

Usage
-----
    >>> from src.lexer import Lexer
    >>> from src.parser import Parser, AssignmentNode, MatrixNode
    >>> tokens = Lexer('A = [1, 2; 3, 4]').tokenize()
    >>> ast = Parser(tokens).parse()
    >>> isinstance(ast, AssignmentNode)
    True
    >>> isinstance(ast.expr, MatrixNode)
    True
    >>> len(ast.expr.rows)
    2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union, cast

from src.lexer import Token, TokenType


# ===========================================================================
# AST node definitions
# ===========================================================================

@dataclass
class NumberNode:
    """A numeric literal node.

    Attributes:
        value: The numeric value. A Python ``float`` for an ordinary numeric
            literal, or a Python ``complex`` (zero real part) for an
            imaginary literal written with a trailing ``i``.

    Example::

        3.14  →  NumberNode(value=3.14)
        42    →  NumberNode(value=42.0)
        3i    →  NumberNode(value=3j)
    """
    value: float | complex


@dataclass
class VarNode:
    """A variable (identifier) reference node.

    Attributes:
        name: The identifier string as it appeared in the source.

    Example::

        A      →  VarNode(name='A')
        my_var →  VarNode(name='my_var')
    """
    name: str


@dataclass
class MatrixNode:
    """A matrix literal node.

    Rows are separated by semicolons; columns within a row are separated by
    commas.  Each element is itself a full ``AST`` expression, which allows
    arithmetic inside matrix literals.

    Attributes:
        rows: List of rows; each row is a list of ``AST`` expression nodes.

    Example::

        [1, 2; 3, 4]  →  MatrixNode(rows=[
                              [NumberNode(1.0), NumberNode(2.0)],
                              [NumberNode(3.0), NumberNode(4.0)],
                          ])
    """
    rows: list[list[AST]]


@dataclass
class BinOpNode:
    """A binary arithmetic operation node.

    Attributes:
        left:  Left-hand side operand expression.
        op:    Operator encoded as the ``TokenType`` member name:
               ``'PLUS'``, ``'MINUS'``, or ``'STAR'``.
        right: Right-hand side operand expression.

    Example::

        A + B  →  BinOpNode(left=VarNode('A'), op='PLUS', right=VarNode('B'))
        A * B  →  BinOpNode(left=VarNode('A'), op='STAR', right=VarNode('B'))
    """
    left: AST
    op: str       # 'PLUS' | 'MINUS' | 'STAR'
    right: AST


@dataclass
class UnaryNode:
    """A unary (prefix) operation node.

    Currently only negation (``-``) is supported.

    Attributes:
        op:      Operator name: ``'MINUS'``.
        operand: The expression being negated.

    Example::

        -A   →  UnaryNode(op='MINUS', operand=VarNode('A'))
        -3   →  UnaryNode(op='MINUS', operand=NumberNode(3.0))
    """
    op: str       # 'MINUS'
    operand: AST


@dataclass
class FunctionNode:
    """A built-in function call node.

    Attributes:
        name: One of the reserved function names:
              ``transpose``, ``det``, ``inv``, ``eye``, ``zeros``, ``ones``,
              ``trace``, ``dagger``, ``outer``, ``tensor``, ``kron``,
              ``commutator``.
        args: Ordered list of argument expressions.

    Example::

        eye(3)         →  FunctionNode(name='eye', args=[NumberNode(3.0)])
        zeros(2, 4)    →  FunctionNode(name='zeros',
                              args=[NumberNode(2.0), NumberNode(4.0)])
        inv(A * B)     →  FunctionNode(name='inv',
                              args=[BinOpNode(VarNode('A'), 'STAR', VarNode('B'))])
    """
    name: str
    args: list[AST]


@dataclass
class AssignmentNode:
    """A variable assignment node.

    Attributes:
        name: Target variable identifier.
        expr: The expression whose evaluated value is stored in the variable.

    Example::

        A = [1, 2]  →  AssignmentNode(name='A',
                            expr=MatrixNode([[NumberNode(1.0), NumberNode(2.0)]]))
    """
    name: str
    expr: AST


# Union type alias representing any valid AST node.
# Defined after all node classes so forward references in field annotations
# (which are lazily evaluated due to ``from __future__ import annotations``)
# are resolvable at inspection time.
AST = Union[
    NumberNode,
    VarNode,
    MatrixNode,
    BinOpNode,
    UnaryNode,
    FunctionNode,
    AssignmentNode,
]


# ===========================================================================
# Parser
# ===========================================================================

class Parser:
    """Recursive descent parser for the LA Shell expression language.

    Accepts a token list produced by ``Lexer.tokenize()`` and builds an AST
    that can be walked by an evaluator.  All grammar rules are implemented as
    private ``parse_*`` methods that call each other recursively.

    The parser never modifies the token list; it only advances an integer
    position cursor (``self.pos``).

    Args:
        tokens: Flat list of Token objects ending with an EOF token.

    Raises:
        SyntaxError: On any grammatical error, trailing tokens, or mismatched
                     brackets.  Error messages include the offending token's
                     value, position, and line number.

    Example::

        >>> from src.lexer import Lexer
        >>> from src.parser import Parser, BinOpNode
        >>> tokens = Lexer('A * B + C').tokenize()
        >>> ast = Parser(tokens).parse()
        >>> isinstance(ast, BinOpNode) and ast.op == 'PLUS'
        True
        >>> isinstance(ast.left, BinOpNode) and ast.left.op == 'STAR'
        True
    """

    def __init__(self, tokens: list[Token]) -> None:
        """Initialise the parser.

        Args:
            tokens: Token list as returned by ``Lexer.tokenize()``.  Must
                    contain at least one token (the EOF sentinel).
        """
        self.tokens: list[Token] = tokens
        self.pos: int = 0

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _peek(self) -> Token:
        """Return the current token without consuming it.

        Returns:
            The token at ``self.pos``, or the last token in the list (which
            is always EOF) if the cursor has reached the end.
        """
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF guard

    def _peek_next(self) -> Token:
        """Return the token one position ahead without consuming it.

        Returns:
            The token at ``self.pos + 1``, or the EOF token if out of range.
        """
        nxt = self.pos + 1
        if nxt < len(self.tokens):
            return self.tokens[nxt]
        return self.tokens[-1]  # EOF guard

    def _consume(self, expected: TokenType | None = None) -> Token:
        """Consume and return the current token.

        Optionally asserts that the current token matches ``expected``.

        Args:
            expected: When provided, the current token must have this type;
                      otherwise a ``SyntaxError`` is raised before consuming.

        Returns:
            The token that was at ``self.pos`` before the call.

        Raises:
            SyntaxError: If ``expected`` is provided and the current token
                         type does not match.
        """
        token = self._peek()
        if expected is not None and token.type != expected:
            raise SyntaxError(
                f"Expected {expected.name} but got "
                f"{token.type.name} ({token.value!r}) "
                f"at position {token.position} (line {token.line})"
            )
        self.pos += 1
        return token

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def parse(self) -> AST:
        """Parse a complete program and return the root AST node.

        Determines whether the input is an assignment or a bare expression by
        inspecting the first two tokens:
        - ``IDENTIFIER`` followed by ``EQUALS`` → assignment
        - anything else → expression

        After the root node is built, any remaining non-EOF tokens are treated
        as a syntax error.

        Returns:
            The root AST node (``AssignmentNode`` or an expression node).

        Raises:
            SyntaxError: On grammatical errors or unexpected trailing tokens.
        """
        if (
            self._peek().type == TokenType.IDENTIFIER
            and self._peek_next().type == TokenType.EQUALS
        ):
            result: AST = self._parse_assignment()
        else:
            result = self._parse_expression()

        # After parsing, only an EOF should remain.
        leftover = self._peek()
        if leftover.type != TokenType.EOF:
            raise SyntaxError(
                f"Unexpected token {leftover.type.name} ({leftover.value!r}) "
                f"at position {leftover.position} (line {leftover.line}) — "
                "trailing input after a complete expression"
            )
        return result

    # -----------------------------------------------------------------------
    # Grammar productions
    # -----------------------------------------------------------------------

    def _parse_assignment(self) -> AssignmentNode:
        """Parse an assignment statement.

        Grammar rule::

            assignment ::= IDENTIFIER '=' expression

        Returns:
            An ``AssignmentNode`` binding the identifier to the parsed
            expression.

        Raises:
            SyntaxError: If the token sequence does not match the rule.
        """
        name_token = self._consume(TokenType.IDENTIFIER)
        self._consume(TokenType.EQUALS)
        expr = self._parse_expression()
        return AssignmentNode(name=str(name_token.value), expr=expr)

    def _parse_expression(self) -> AST:
        """Parse an additive expression.

        Grammar rule::

            expression ::= term (('+' | '-') term)*

        Implements left-associative addition and subtraction at the lowest
        precedence level (below multiplication).

        Returns:
            An AST node for the complete additive expression.  If no ``+``
            or ``-`` operators are present, the node from ``_parse_term``
            is returned directly without wrapping.

        Raises:
            SyntaxError: On grammatical errors within sub-expressions.
        """
        left: AST = self._parse_term()
        while self._peek().type in (TokenType.PLUS, TokenType.MINUS):
            op_token = self._consume()          # consume + or -
            right: AST = self._parse_term()
            left = BinOpNode(left=left, op=op_token.type.name, right=right)
        return left

    def _parse_term(self) -> AST:
        """Parse a multiplicative expression.

        Grammar rule::

            term ::= factor ('*' factor)*

        Implements left-associative multiplication at higher precedence than
        additive operators.  The ``STAR`` operator maps to NumPy's ``@``
        (matrix multiplication) in the evaluator.

        Returns:
            An AST node for the complete multiplicative expression.  If no
            ``*`` operators are present, the node from ``_parse_factor``
            is returned directly.

        Raises:
            SyntaxError: On grammatical errors within sub-expressions.
        """
        left: AST = self._parse_factor()
        while self._peek().type == TokenType.STAR:
            self._consume(TokenType.STAR)
            right: AST = self._parse_factor()
            left = BinOpNode(left=left, op='STAR', right=right)
        return left

    def _parse_factor(self) -> AST:
        """Parse an atomic factor or unary prefix expression.

        Grammar rule::

            factor ::= NUMBER
                     | FUNCTION '(' expression (',' expression)* ')'
                     | IDENTIFIER
                     | '[' row (';' row)* ']'
                     | '-' factor
                     | '(' expression ')'

        This method dispatches on the type of the current token and delegates
        to specialised helpers for matrix literals and function calls.
        Unary negation is handled here with right-associativity: ``--A``
        is valid and equivalent to ``A``.

        Returns:
            The AST node for the factor.

        Raises:
            SyntaxError: If the current token cannot start a valid factor.
        """
        token = self._peek()

        # ---- Numeric literal ----
        if token.type == TokenType.NUMBER:
            self._consume()
            return NumberNode(value=cast(float, token.value))  # Lexer already gives float

        # ---- Imaginary literal (e.g. 3i, 2.5i) ----
        if token.type == TokenType.IMAGINARY:
            self._consume()
            return NumberNode(value=cast(complex, token.value))  # Lexer already gives complex

        # ---- Built-in function call ----
        # Must be checked before IDENTIFIER because function names are emitted
        # as FUNCTION tokens by the Lexer, never as IDENTIFIER tokens.
        if token.type == TokenType.FUNCTION:
            return self._parse_function_call()

        # ---- Variable reference ----
        if token.type == TokenType.IDENTIFIER:
            self._consume()
            return VarNode(name=str(token.value))

        # ---- Matrix literal ----
        if token.type == TokenType.LBRACKET:
            return self._parse_matrix()

        # ---- Unary negation (right-associative via recursion) ----
        if token.type == TokenType.MINUS:
            self._consume(TokenType.MINUS)
            operand = self._parse_factor()
            return UnaryNode(op='MINUS', operand=operand)

        # ---- Parenthesised sub-expression ----
        if token.type == TokenType.LPAREN:
            self._consume(TokenType.LPAREN)
            expr = self._parse_expression()
            self._consume(TokenType.RPAREN)
            return expr

        # ---- Nothing matched ----
        raise SyntaxError(
            f"Unexpected token {token.type.name} ({token.value!r}) "
            f"at position {token.position} (line {token.line}); "
            "expected a number, identifier, '[', '(', '-', or a function name"
        )

    def _parse_matrix(self) -> MatrixNode:
        """Parse a matrix literal.

        Grammar rule::

            matrix ::= '[' row (';' row)* ']'

        Rows are separated by semicolons (MATLAB style).  Each element within
        a row is a full expression, so arithmetic and nested matrices are
        allowed inside matrix literals.

        Returns:
            A ``MatrixNode`` whose ``rows`` attribute is a list of rows,
            each row being a list of AST expression nodes.

        Raises:
            SyntaxError: If the opening bracket has no matching closing
                         bracket, or if a row is syntactically invalid.
        """
        open_tok = self._consume(TokenType.LBRACKET)
        rows: list[list[AST]] = [self._parse_row()]

        while self._peek().type == TokenType.SEMICOLON:
            self._consume(TokenType.SEMICOLON)
            rows.append(self._parse_row())

        closing = self._peek()
        if closing.type != TokenType.RBRACKET:
            raise SyntaxError(
                f"Mismatched brackets: expected ']' but found "
                f"{closing.type.name} ({closing.value!r}) "
                f"at position {closing.position} (line {closing.line}). "
                f"Opening '[' was at position {open_tok.position} "
                f"(line {open_tok.line})"
            )
        self._consume(TokenType.RBRACKET)
        return MatrixNode(rows=rows)

    def _parse_row(self) -> list[AST]:
        """Parse a single matrix row.

        Grammar rule::

            row ::= expression (',' expression)*

        Returns:
            An ordered list of AST expression nodes, one per column element.

        Raises:
            SyntaxError: On grammatical errors in element expressions.
        """
        elements: list[AST] = [self._parse_expression()]
        while self._peek().type == TokenType.COMMA:
            self._consume(TokenType.COMMA)
            elements.append(self._parse_expression())
        return elements

    def _parse_function_call(self) -> FunctionNode:
        """Parse a built-in function call.

        Grammar rule::

            function_call ::= FUNCTION '(' expression (',' expression)* ')'

        Arguments are full expressions; e.g. ``inv(A * B)`` and
        ``zeros(n + 1, m)`` are both valid.

        Returns:
            A ``FunctionNode`` with the function name and argument list.

        Raises:
            SyntaxError: If the argument list or parentheses are malformed.
        """
        func_tok = self._consume(TokenType.FUNCTION)
        self._consume(TokenType.LPAREN)
        args: list[AST] = [self._parse_expression()]
        while self._peek().type == TokenType.COMMA:
            self._consume(TokenType.COMMA)
            args.append(self._parse_expression())
        self._consume(TokenType.RPAREN)
        return FunctionNode(name=str(func_tok.value), args=args)

    # -----------------------------------------------------------------------
    # Public convenience aliases (match names used in CONTEXT.md spec)
    # -----------------------------------------------------------------------

    def parse_expression(self) -> AST:
        """Public alias for ``_parse_expression``.

        Useful when the caller wants to invoke the expression-level rule
        directly (e.g. in tests or a sub-expression evaluator) without
        going through the full ``parse()`` entry point.

        Returns:
            The root AST node of the parsed expression.
        """
        return self._parse_expression()

    def parse_term(self) -> AST:
        """Public alias for ``_parse_term``.

        Returns:
            The root AST node of the parsed term.
        """
        return self._parse_term()

    def parse_factor(self) -> AST:
        """Public alias for ``_parse_factor``.

        Returns:
            The root AST node of the parsed factor.
        """
        return self._parse_factor()

    def parse_matrix(self) -> MatrixNode:
        """Public alias for ``_parse_matrix``.

        Returns:
            The ``MatrixNode`` for the parsed matrix literal.
        """
        return self._parse_matrix()

    def parse_function_call(self) -> FunctionNode:
        """Public alias for ``_parse_function_call``.

        Returns:
            The ``FunctionNode`` for the parsed function call.
        """
        return self._parse_function_call()
