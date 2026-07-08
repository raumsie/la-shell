"""Tests for src.parser — recursive descent parser and AST nodes."""
from __future__ import annotations

import pytest

from src.lexer import Lexer
from src.parser import (
    AssignmentNode,
    BinOpNode,
    FunctionNode,
    MatrixNode,
    NumberNode,
    Parser,
    UnaryNode,
    VarNode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse(source: str):
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


# ---------------------------------------------------------------------------
# NumberNode
# ---------------------------------------------------------------------------

class TestNumberNode:
    def test_integer_produces_number_node(self):
        node = parse("3")
        assert isinstance(node, NumberNode)

    def test_number_node_value_is_float(self):
        node = parse("42")
        assert node.value == 42.0
        assert isinstance(node.value, float)

    def test_float_literal_value(self):
        node = parse("3.14")
        assert abs(node.value - 3.14) < 1e-9

    def test_leading_dot_float_value(self):
        node = parse(".5")
        assert abs(node.value - 0.5) < 1e-9

    def test_zero(self):
        node = parse("0")
        assert node.value == 0.0


# ---------------------------------------------------------------------------
# NumberNode — imaginary literals
# ---------------------------------------------------------------------------

class TestImaginaryLiteral:
    def test_imaginary_literal_produces_number_node(self):
        node = parse("3i")
        assert isinstance(node, NumberNode)

    def test_imaginary_literal_value_is_complex(self):
        node = parse("3i")
        assert node.value == complex(0, 3)
        assert isinstance(node.value, complex)

    def test_float_imaginary_literal_value(self):
        node = parse("2.5i")
        assert node.value == complex(0, 2.5)

    def test_compound_complex_expression_is_binop(self):
        node = parse("3+4i")
        assert isinstance(node, BinOpNode)
        assert node.op == "PLUS"
        assert isinstance(node.left, NumberNode) and node.left.value == 3.0
        assert isinstance(node.right, NumberNode) and node.right.value == complex(0, 4)

    def test_bare_i_parses_as_var_node(self):
        node = parse("i")
        assert isinstance(node, VarNode)
        assert node.name == "i"


# ---------------------------------------------------------------------------
# VarNode
# ---------------------------------------------------------------------------

class TestVarNode:
    def test_identifier_produces_var_node(self):
        node = parse("A")
        assert isinstance(node, VarNode)

    def test_var_node_name(self):
        node = parse("myVar")
        assert node.name == "myVar"

    def test_underscore_identifier(self):
        node = parse("my_var")
        assert isinstance(node, VarNode)
        assert node.name == "my_var"


# ---------------------------------------------------------------------------
# AssignmentNode
# ---------------------------------------------------------------------------

class TestAssignmentNode:
    def test_simple_assignment_produces_assignment_node(self):
        node = parse("A = 3")
        assert isinstance(node, AssignmentNode)

    def test_assignment_name(self):
        node = parse("A = 3")
        assert node.name == "A"

    def test_assignment_expr_is_number_node(self):
        node = parse("A = 3")
        assert isinstance(node.expr, NumberNode)
        assert node.expr.value == 3.0

    def test_assignment_expr_is_var_node(self):
        node = parse("B = A")
        assert isinstance(node.expr, VarNode)
        assert node.expr.name == "A"

    def test_assignment_with_matrix_rhs(self):
        node = parse("A = [1, 2; 3, 4]")
        assert isinstance(node, AssignmentNode)
        assert isinstance(node.expr, MatrixNode)

    def test_assignment_name_exactly_matches_source(self):
        node = parse("Result = 42")
        assert node.name == "Result"


# ---------------------------------------------------------------------------
# MatrixNode
# ---------------------------------------------------------------------------

class TestMatrixNode:
    def test_single_row_matrix(self):
        node = parse("[1, 2, 3]")
        assert isinstance(node, MatrixNode)
        assert len(node.rows) == 1
        assert len(node.rows[0]) == 3

    def test_two_row_matrix(self):
        node = parse("[1, 2; 3, 4]")
        assert isinstance(node, MatrixNode)
        assert len(node.rows) == 2

    def test_two_row_matrix_col_count(self):
        node = parse("[1, 2; 3, 4]")
        assert len(node.rows[0]) == 2
        assert len(node.rows[1]) == 2

    def test_matrix_elements_are_number_nodes(self):
        node = parse("[1, 2]")
        assert isinstance(node.rows[0][0], NumberNode)
        assert isinstance(node.rows[0][1], NumberNode)

    def test_matrix_element_values(self):
        node = parse("[1, 2; 3, 4]")
        assert node.rows[0][0].value == 1.0
        assert node.rows[0][1].value == 2.0
        assert node.rows[1][0].value == 3.0
        assert node.rows[1][1].value == 4.0

    def test_single_element_matrix(self):
        node = parse("[5]")
        assert isinstance(node, MatrixNode)
        assert len(node.rows) == 1
        assert len(node.rows[0]) == 1

    def test_three_row_matrix(self):
        node = parse("[1; 2; 3]")
        assert len(node.rows) == 3

    def test_matrix_element_can_be_expression(self):
        node = parse("[1 + 2, 3]")
        assert isinstance(node.rows[0][0], BinOpNode)

    def test_1x3_matrix(self):
        node = parse("[1, 2, 3]")
        assert len(node.rows) == 1
        assert len(node.rows[0]) == 3


# ---------------------------------------------------------------------------
# BinOpNode — operator precedence
# ---------------------------------------------------------------------------

class TestOperatorPrecedence:
    def test_addition_produces_binop_plus(self):
        node = parse("1 + 2")
        assert isinstance(node, BinOpNode)
        assert node.op == "PLUS"

    def test_subtraction_produces_binop_minus(self):
        node = parse("1 - 2")
        assert isinstance(node, BinOpNode)
        assert node.op == "MINUS"

    def test_multiplication_produces_binop_star(self):
        node = parse("A * B")
        assert isinstance(node, BinOpNode)
        assert node.op == "STAR"

    def test_multiplication_binds_tighter_than_addition(self):
        # 1 + 2 * 3  must parse as  1 + (2 * 3)
        node = parse("1 + 2 * 3")
        assert isinstance(node, BinOpNode)
        assert node.op == "PLUS"
        assert isinstance(node.left, NumberNode)
        assert node.left.value == 1.0
        assert isinstance(node.right, BinOpNode)
        assert node.right.op == "STAR"

    def test_star_right_operands_correct(self):
        node = parse("1 + 2 * 3")
        inner = node.right
        assert inner.left.value == 2.0
        assert inner.right.value == 3.0

    def test_multiplication_before_addition_complex(self):
        # A * B + C → BinOpNode(PLUS, BinOpNode(STAR, A, B), C)
        node = parse("A * B + C")
        assert node.op == "PLUS"
        assert node.left.op == "STAR"

    def test_subtraction_before_add_complex(self):
        node = parse("A + B * C - D")
        assert node.op == "MINUS"
        assert node.left.op == "PLUS"


# ---------------------------------------------------------------------------
# Left-associativity
# ---------------------------------------------------------------------------

class TestLeftAssociativity:
    def test_subtraction_is_left_associative(self):
        # 1 - 2 - 3 must parse as (1 - 2) - 3
        node = parse("1 - 2 - 3")
        assert node.op == "MINUS"
        assert isinstance(node.left, BinOpNode)
        assert node.left.op == "MINUS"
        assert node.left.left.value == 1.0
        assert node.left.right.value == 2.0
        assert node.right.value == 3.0

    def test_addition_is_left_associative(self):
        node = parse("1 + 2 + 3")
        assert node.op == "PLUS"
        assert isinstance(node.left, BinOpNode)
        assert node.left.op == "PLUS"

    def test_multiplication_is_left_associative(self):
        node = parse("A * B * C")
        assert node.op == "STAR"
        assert isinstance(node.left, BinOpNode)
        assert node.left.op == "STAR"


# ---------------------------------------------------------------------------
# UnaryNode
# ---------------------------------------------------------------------------

class TestUnaryNode:
    def test_unary_minus_identifier(self):
        node = parse("-A")
        assert isinstance(node, UnaryNode)
        assert node.op == "MINUS"
        assert isinstance(node.operand, VarNode)
        assert node.operand.name == "A"

    def test_unary_minus_number(self):
        node = parse("-3")
        assert isinstance(node, UnaryNode)
        assert node.op == "MINUS"
        assert isinstance(node.operand, NumberNode)
        assert node.operand.value == 3.0

    def test_double_unary_minus(self):
        # --A is valid: UnaryNode(MINUS, UnaryNode(MINUS, VarNode('A')))
        node = parse("--A")
        assert isinstance(node, UnaryNode)
        assert isinstance(node.operand, UnaryNode)

    def test_unary_minus_matrix(self):
        node = parse("-[1, 2]")
        assert isinstance(node, UnaryNode)
        assert isinstance(node.operand, MatrixNode)


# ---------------------------------------------------------------------------
# FunctionNode
# ---------------------------------------------------------------------------

class TestFunctionNode:
    def test_det_single_arg(self):
        node = parse("det(A)")
        assert isinstance(node, FunctionNode)
        assert node.name == "det"
        assert len(node.args) == 1
        assert isinstance(node.args[0], VarNode)
        assert node.args[0].name == "A"

    def test_zeros_two_args(self):
        node = parse("zeros(2, 3)")
        assert isinstance(node, FunctionNode)
        assert node.name == "zeros"
        assert len(node.args) == 2
        assert node.args[0].value == 2.0
        assert node.args[1].value == 3.0

    def test_eye_single_arg(self):
        node = parse("eye(3)")
        assert isinstance(node, FunctionNode)
        assert node.name == "eye"
        assert len(node.args) == 1
        assert node.args[0].value == 3.0

    def test_transpose_single_arg(self):
        node = parse("transpose(A)")
        assert isinstance(node, FunctionNode)
        assert node.name == "transpose"
        assert len(node.args) == 1

    def test_inv_single_arg(self):
        node = parse("inv(A)")
        assert isinstance(node, FunctionNode)
        assert node.name == "inv"

    def test_ones_two_args(self):
        node = parse("ones(2, 3)")
        assert isinstance(node, FunctionNode)
        assert node.name == "ones"
        assert len(node.args) == 2

    def test_trace_single_arg(self):
        node = parse("trace(A)")
        assert isinstance(node, FunctionNode)
        assert node.name == "trace"

    def test_function_arg_can_be_expression(self):
        node = parse("inv(A * B)")
        assert isinstance(node, FunctionNode)
        assert isinstance(node.args[0], BinOpNode)
        assert node.args[0].op == "STAR"


# ---------------------------------------------------------------------------
# Parenthesised expressions
# ---------------------------------------------------------------------------

class TestParenthesisedExpressions:
    def test_parens_change_precedence(self):
        # (A + B) * C → STAR, not PLUS at root
        node = parse("(A + B) * C")
        assert isinstance(node, BinOpNode)
        assert node.op == "STAR"

    def test_parens_left_operand_is_plus(self):
        node = parse("(A + B) * C")
        assert node.left.op == "PLUS"

    def test_redundant_parens_dont_wrap_extra_node(self):
        node = parse("(A)")
        # Just a VarNode — no extra wrapping
        assert isinstance(node, VarNode)

    def test_nested_parens(self):
        node = parse("((3))")
        assert isinstance(node, NumberNode)
        assert node.value == 3.0


# ---------------------------------------------------------------------------
# SyntaxError cases
# ---------------------------------------------------------------------------

class TestSyntaxErrors:
    def test_unclosed_bracket_raises_syntax_error(self):
        with pytest.raises(SyntaxError):
            parse("[1, 2")

    def test_missing_rparen_raises_syntax_error(self):
        with pytest.raises(SyntaxError):
            parse("det(A")

    def test_trailing_tokens_raise_syntax_error(self):
        with pytest.raises(SyntaxError):
            parse("1 + 2 3")

    def test_empty_input_raises_syntax_error(self):
        with pytest.raises(SyntaxError):
            parse("")

    def test_only_operator_raises_syntax_error(self):
        with pytest.raises(SyntaxError):
            parse("+")

    def test_double_equals_raises_syntax_error(self):
        with pytest.raises(SyntaxError):
            parse("A == 3")

    def test_unclosed_paren_raises_syntax_error(self):
        with pytest.raises(SyntaxError):
            parse("(1 + 2")

    def test_missing_rhs_of_assignment_raises_syntax_error(self):
        with pytest.raises(SyntaxError):
            parse("A =")

    def test_missing_rhs_of_addition_raises_syntax_error(self):
        with pytest.raises(SyntaxError):
            parse("A +")
