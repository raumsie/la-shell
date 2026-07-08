"""Tests for src.lexer — Lexer and TokenType."""
from __future__ import annotations

import pytest

from src.lexer import Lexer, LexerError, TokenType, FUNCTION_KEYWORDS, Token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize(source: str) -> list[Token]:
    return Lexer(source).tokenize()


def types(source: str) -> list[str]:
    return [t.type.name for t in tokenize(source)]


def values(source: str) -> list:
    return [t.value for t in tokenize(source)]


# ---------------------------------------------------------------------------
# TokenType completeness
# ---------------------------------------------------------------------------

class TestTokenTypeMembers:
    """All 15 TokenType members must exist."""

    EXPECTED_MEMBERS = [
        "NUMBER", "IMAGINARY", "IDENTIFIER", "LBRACKET", "RBRACKET",
        "SEMICOLON", "COMMA", "EQUALS", "PLUS", "MINUS", "STAR", "LPAREN",
        "RPAREN", "FUNCTION", "DOT", "EOF",
    ]

    def test_all_16_members_exist(self):
        names = [m.name for m in TokenType]
        for name in self.EXPECTED_MEMBERS:
            assert name in names, f"Missing TokenType member: {name}"

    def test_exactly_16_members(self):
        assert len(TokenType) == 16


# ---------------------------------------------------------------------------
# EOF token
# ---------------------------------------------------------------------------

class TestEOF:
    def test_eof_is_last_token(self):
        tokens = tokenize("1 + 2")
        assert tokens[-1].type == TokenType.EOF

    def test_eof_on_empty_string(self):
        tokens = tokenize("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_eof_value_is_empty_string(self):
        tokens = tokenize("42")
        assert tokens[-1].value == ""

    def test_eof_position_equals_source_length(self):
        source = "1 + 2"
        tokens = tokenize(source)
        assert tokens[-1].position == len(source)


# ---------------------------------------------------------------------------
# NUMBER tokens
# ---------------------------------------------------------------------------

class TestNumberTokens:
    def test_integer_produces_number_token(self):
        toks = tokenize("3")
        assert toks[0].type == TokenType.NUMBER

    def test_integer_value_is_float(self):
        toks = tokenize("3")
        assert toks[0].value == 3.0
        assert isinstance(toks[0].value, float)

    def test_float_with_decimal(self):
        toks = tokenize("3.14")
        assert toks[0].type == TokenType.NUMBER
        assert abs(toks[0].value - 3.14) < 1e-9

    def test_leading_dot_float(self):
        toks = tokenize(".5")
        assert toks[0].type == TokenType.NUMBER
        assert abs(toks[0].value - 0.5) < 1e-9

    def test_trailing_dot_float(self):
        toks = tokenize("1.")
        assert toks[0].type == TokenType.NUMBER
        assert toks[0].value == 1.0

    def test_zero_integer(self):
        toks = tokenize("0")
        assert toks[0].value == 0.0

    def test_large_integer(self):
        toks = tokenize("99999")
        assert toks[0].value == 99999.0

    def test_multi_digit_float(self):
        toks = tokenize("123.456")
        assert abs(toks[0].value - 123.456) < 1e-9


# ---------------------------------------------------------------------------
# IDENTIFIER vs FUNCTION tokens
# ---------------------------------------------------------------------------

class TestIdentifierAndFunctionTokens:
    def test_plain_identifier_produces_identifier_token(self):
        toks = tokenize("A")
        assert toks[0].type == TokenType.IDENTIFIER

    def test_identifier_value_is_string(self):
        toks = tokenize("myVar")
        assert toks[0].value == "myVar"

    def test_multi_char_identifier(self):
        toks = tokenize("my_var")
        assert toks[0].type == TokenType.IDENTIFIER
        assert toks[0].value == "my_var"

    def test_identifier_starting_with_underscore(self):
        toks = tokenize("_x")
        assert toks[0].type == TokenType.IDENTIFIER

    def test_identifier_with_digits(self):
        toks = tokenize("x1")
        assert toks[0].type == TokenType.IDENTIFIER

    def test_all_7_function_keywords_produce_function_token(self):
        for kw in FUNCTION_KEYWORDS:
            toks = tokenize(kw)
            assert toks[0].type == TokenType.FUNCTION, f"{kw!r} should be FUNCTION"

    def test_transpose_is_function(self):
        assert tokenize("transpose")[0].type == TokenType.FUNCTION

    def test_det_is_function(self):
        assert tokenize("det")[0].type == TokenType.FUNCTION

    def test_inv_is_function(self):
        assert tokenize("inv")[0].type == TokenType.FUNCTION

    def test_eye_is_function(self):
        assert tokenize("eye")[0].type == TokenType.FUNCTION

    def test_zeros_is_function(self):
        assert tokenize("zeros")[0].type == TokenType.FUNCTION

    def test_ones_is_function(self):
        assert tokenize("ones")[0].type == TokenType.FUNCTION

    def test_trace_is_function(self):
        assert tokenize("trace")[0].type == TokenType.FUNCTION

    def test_dagger_is_function(self):
        assert tokenize("dagger")[0].type == TokenType.FUNCTION

    def test_outer_is_function(self):
        assert tokenize("outer")[0].type == TokenType.FUNCTION

    def test_tensor_is_function(self):
        assert tokenize("tensor")[0].type == TokenType.FUNCTION

    def test_kron_is_function(self):
        assert tokenize("kron")[0].type == TokenType.FUNCTION

    def test_commutator_is_function(self):
        assert tokenize("commutator")[0].type == TokenType.FUNCTION

    def test_function_keyword_value_is_name_string(self):
        toks = tokenize("det")
        assert toks[0].value == "det"

    def test_non_keyword_word_is_identifier_not_function(self):
        toks = tokenize("deter")
        assert toks[0].type == TokenType.IDENTIFIER

    def test_function_keywords_are_exactly_12(self):
        assert len(FUNCTION_KEYWORDS) == 12


# ---------------------------------------------------------------------------
# Single-character tokens
# ---------------------------------------------------------------------------

class TestSingleCharacterTokens:
    def test_lbracket(self):
        assert tokenize("[")[0].type == TokenType.LBRACKET

    def test_rbracket(self):
        assert tokenize("]")[0].type == TokenType.RBRACKET

    def test_semicolon(self):
        assert tokenize(";")[0].type == TokenType.SEMICOLON

    def test_comma(self):
        assert tokenize(",")[0].type == TokenType.COMMA

    def test_equals(self):
        assert tokenize("=")[0].type == TokenType.EQUALS

    def test_plus(self):
        assert tokenize("+")[0].type == TokenType.PLUS

    def test_minus(self):
        assert tokenize("-")[0].type == TokenType.MINUS

    def test_star(self):
        assert tokenize("*")[0].type == TokenType.STAR

    def test_lparen(self):
        assert tokenize("(")[0].type == TokenType.LPAREN

    def test_rparen(self):
        assert tokenize(")")[0].type == TokenType.RPAREN

    def test_dot(self):
        assert tokenize(".")[0].type == TokenType.DOT

    def test_single_char_value_is_the_character(self):
        assert tokenize("[")[0].value == "["
        assert tokenize("]")[0].value == "]"
        assert tokenize(";")[0].value == ";"
        assert tokenize(",")[0].value == ","
        assert tokenize("=")[0].value == "="
        assert tokenize("+")[0].value == "+"
        assert tokenize("-")[0].value == "-"
        assert tokenize("*")[0].value == "*"
        assert tokenize("(")[0].value == "("
        assert tokenize(")")[0].value == ")"
        assert tokenize(".")[0].value == "."


# ---------------------------------------------------------------------------
# Whitespace and newlines
# ---------------------------------------------------------------------------

class TestWhitespaceAndNewlines:
    def test_spaces_are_skipped(self):
        toks = tokenize("  1  ")
        assert len(toks) == 2  # NUMBER + EOF
        assert toks[0].type == TokenType.NUMBER

    def test_tabs_are_skipped(self):
        toks = tokenize("\t1\t")
        assert toks[0].type == TokenType.NUMBER

    def test_newline_produces_no_token(self):
        toks = tokenize("1\n2")
        typs = [t.type for t in toks]
        assert TokenType.NUMBER in typs
        # Only NUMBERs and EOF — no NEWLINE token
        for t in toks:
            assert t.type in (TokenType.NUMBER, TokenType.EOF)

    def test_newline_increments_line_counter(self):
        toks = tokenize("1\n2")
        # First NUMBER is on line 1; second NUMBER is on line 2
        num_tokens = [t for t in toks if t.type == TokenType.NUMBER]
        assert num_tokens[0].line == 1
        assert num_tokens[1].line == 2

    def test_multiple_newlines_accumulate_line_count(self):
        toks = tokenize("1\n\n\n2")
        num_tokens = [t for t in toks if t.type == TokenType.NUMBER]
        assert num_tokens[0].line == 1
        assert num_tokens[1].line == 4


# ---------------------------------------------------------------------------
# LexerError for unrecognised characters
# ---------------------------------------------------------------------------

class TestLexerError:
    def test_at_sign_raises_lexer_error(self):
        with pytest.raises(LexerError):
            tokenize("@")

    def test_hash_raises_lexer_error(self):
        with pytest.raises(LexerError):
            tokenize("#")

    def test_dollar_sign_raises_lexer_error(self):
        with pytest.raises(LexerError):
            tokenize("$")

    def test_exclamation_raises_lexer_error(self):
        with pytest.raises(LexerError):
            tokenize("!")

    def test_ampersand_raises_lexer_error(self):
        with pytest.raises(LexerError):
            tokenize("&")

    def test_lexer_error_carries_position(self):
        with pytest.raises(LexerError) as exc_info:
            tokenize("1 @ 2")
        assert exc_info.value.position == 2

    def test_lexer_error_carries_line(self):
        with pytest.raises(LexerError) as exc_info:
            tokenize("@")
        assert exc_info.value.line == 1

    def test_lexer_error_line_after_newline(self):
        with pytest.raises(LexerError) as exc_info:
            tokenize("1\n@")
        assert exc_info.value.line == 2

    def test_lexer_error_message_contains_character(self):
        with pytest.raises(LexerError, match="@"):
            tokenize("@")


# ---------------------------------------------------------------------------
# Token position tracking
# ---------------------------------------------------------------------------

class TestTokenPositions:
    def test_first_token_position_is_zero(self):
        toks = tokenize("A")
        assert toks[0].position == 0

    def test_token_position_after_space(self):
        toks = tokenize("A B")
        assert toks[1].position == 2

    def test_line_starts_at_one(self):
        toks = tokenize("A")
        assert toks[0].line == 1


# ---------------------------------------------------------------------------
# IMAGINARY tokens
# ---------------------------------------------------------------------------

class TestImaginaryTokens:
    def test_integer_imaginary_produces_imaginary_token(self):
        toks = tokenize("3i")
        assert toks[0].type == TokenType.IMAGINARY

    def test_integer_imaginary_value_is_complex(self):
        toks = tokenize("3i")
        assert toks[0].value == complex(0, 3)

    def test_float_imaginary_value(self):
        toks = tokenize("2.5i")
        assert toks[0].type == TokenType.IMAGINARY
        assert toks[0].value == complex(0, 2.5)

    def test_leading_dot_imaginary_value(self):
        toks = tokenize(".5i")
        assert toks[0].type == TokenType.IMAGINARY
        assert toks[0].value == complex(0, 0.5)

    def test_imaginary_token_consumes_single_token(self):
        # "3i" must be one IMAGINARY token, not NUMBER followed by IDENTIFIER.
        toks = tokenize("3i")
        assert len(toks) == 2  # IMAGINARY + EOF

    def test_bare_i_is_identifier_not_imaginary(self):
        # No leading digit means 'i' stays an ordinary identifier, so it
        # remains usable as a variable name.
        toks = tokenize("i")
        assert toks[0].type == TokenType.IDENTIFIER
        assert toks[0].value == "i"

    def test_identifier_starting_with_i_unaffected(self):
        toks = tokenize("i_var")
        assert toks[0].type == TokenType.IDENTIFIER
        assert toks[0].value == "i_var"

    def test_inv_still_lexes_as_function(self):
        # Regression guard: the imaginary-literal regex must not interfere
        # with words that happen to start with 'i'.
        assert tokenize("inv")[0].type == TokenType.FUNCTION

    def test_compound_expression_token_types(self):
        result = types("3+4i")
        assert result == ["NUMBER", "PLUS", "IMAGINARY", "EOF"]

    def test_compound_expression_with_minus(self):
        result = types("3-4i")
        assert result == ["NUMBER", "MINUS", "IMAGINARY", "EOF"]


# ---------------------------------------------------------------------------
# Matrix literal tokenization
# ---------------------------------------------------------------------------

class TestMatrixLiteralTokenization:
    def test_simple_matrix_token_types(self):
        result = types("[1, 2; 3, 4]")
        assert result == [
            "LBRACKET", "NUMBER", "COMMA", "NUMBER",
            "SEMICOLON", "NUMBER", "COMMA", "NUMBER",
            "RBRACKET", "EOF",
        ]

    def test_matrix_token_values(self):
        toks = tokenize("[1, 2]")
        assert toks[1].value == 1.0
        assert toks[3].value == 2.0

    def test_matrix_with_floats(self):
        toks = tokenize("[1.5, 2.5]")
        assert abs(toks[1].value - 1.5) < 1e-9
        assert abs(toks[3].value - 2.5) < 1e-9


# ---------------------------------------------------------------------------
# Full expression tokenization
# ---------------------------------------------------------------------------

class TestExpressionTokenization:
    def test_assignment_token_types(self):
        result = types("A = 3")
        assert result == ["IDENTIFIER", "EQUALS", "NUMBER", "EOF"]

    def test_addition_token_types(self):
        result = types("A + B")
        assert result == ["IDENTIFIER", "PLUS", "IDENTIFIER", "EOF"]

    def test_function_call_token_types(self):
        result = types("det(A)")
        assert result == ["FUNCTION", "LPAREN", "IDENTIFIER", "RPAREN", "EOF"]

    def test_dot_command_token_types(self):
        result = types(".help")
        assert result == ["DOT", "IDENTIFIER", "EOF"]

    def test_multiplication_token_types(self):
        result = types("A * B")
        assert result == ["IDENTIFIER", "STAR", "IDENTIFIER", "EOF"]

    def test_unary_minus_token_types(self):
        result = types("-A")
        assert result == ["MINUS", "IDENTIFIER", "EOF"]

    def test_parenthesised_expression_token_types(self):
        result = types("(A + B)")
        assert result == [
            "LPAREN", "IDENTIFIER", "PLUS", "IDENTIFIER", "RPAREN", "EOF"
        ]
