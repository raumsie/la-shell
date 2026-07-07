"""Tests for src.evaluator — Evaluator."""
from __future__ import annotations

import pytest
import numpy as np

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
from src.symbol_table import SymbolTable
from src.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def st() -> SymbolTable:
    return SymbolTable()


@pytest.fixture
def ev(st) -> Evaluator:
    return Evaluator(st)


def eval_source(source: str, evaluator: Evaluator) -> np.ndarray:
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    return evaluator.eval(ast)


# ---------------------------------------------------------------------------
# NumberNode
# ---------------------------------------------------------------------------

class TestEvalNumberNode:
    def test_number_node_returns_ndarray(self, ev):
        result = ev.eval(NumberNode(3.0))
        assert isinstance(result, np.ndarray)

    def test_number_node_shape_is_1x1(self, ev):
        result = ev.eval(NumberNode(3.0))
        assert result.shape == (1, 1)

    def test_number_node_value_correct(self, ev):
        result = ev.eval(NumberNode(7.0))
        assert result.item() == pytest.approx(7.0)

    def test_number_node_zero(self, ev):
        result = ev.eval(NumberNode(0.0))
        assert result.item() == 0.0

    def test_number_node_negative_value(self, ev):
        result = ev.eval(NumberNode(-5.0))
        assert result.item() == pytest.approx(-5.0)

    def test_number_node_float_value(self, ev):
        result = ev.eval(NumberNode(3.14))
        assert result.item() == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# VarNode
# ---------------------------------------------------------------------------

class TestEvalVarNode:
    def test_var_node_returns_stored_array(self, ev, st):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        st.set("A", arr)
        result = ev.eval(VarNode("A"))
        np.testing.assert_array_equal(result, arr)

    def test_var_node_undefined_raises_name_error(self, ev):
        with pytest.raises(NameError):
            ev.eval(VarNode("Undefined"))

    def test_var_node_name_error_message(self, ev):
        with pytest.raises(NameError, match="Undefined"):
            ev.eval(VarNode("Undefined"))

    def test_var_node_after_assignment(self, ev, st):
        st.set("X", np.array([[99.0]]))
        result = ev.eval(VarNode("X"))
        assert result.item() == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# MatrixNode
# ---------------------------------------------------------------------------

class TestEvalMatrixNode:
    def test_1x1_matrix(self, ev):
        node = MatrixNode(rows=[[NumberNode(5.0)]])
        result = ev.eval(node)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(5.0)

    def test_2x2_matrix_shape(self, ev):
        node = MatrixNode(rows=[
            [NumberNode(1.0), NumberNode(2.0)],
            [NumberNode(3.0), NumberNode(4.0)],
        ])
        result = ev.eval(node)
        assert result.shape == (2, 2)

    def test_2x2_matrix_values(self, ev):
        node = MatrixNode(rows=[
            [NumberNode(1.0), NumberNode(2.0)],
            [NumberNode(3.0), NumberNode(4.0)],
        ])
        result = ev.eval(node)
        np.testing.assert_array_almost_equal(
            result, np.array([[1.0, 2.0], [3.0, 4.0]])
        )

    def test_1x3_matrix_shape(self, ev):
        node = MatrixNode(rows=[[NumberNode(1.0), NumberNode(2.0), NumberNode(3.0)]])
        result = ev.eval(node)
        assert result.shape == (1, 3)

    def test_1x3_matrix_values(self, ev):
        result = ev.eval(
            MatrixNode(rows=[[NumberNode(1.0), NumberNode(2.0), NumberNode(3.0)]])
        )
        np.testing.assert_array_almost_equal(result, np.array([[1.0, 2.0, 3.0]]))

    def test_3x1_matrix_shape(self, ev):
        node = MatrixNode(rows=[
            [NumberNode(1.0)],
            [NumberNode(2.0)],
            [NumberNode(3.0)],
        ])
        result = ev.eval(node)
        assert result.shape == (3, 1)

    def test_matrix_dtype_is_float(self, ev):
        node = MatrixNode(rows=[[NumberNode(1.0)]])
        result = ev.eval(node)
        assert result.dtype == np.float64

    def test_matrix_cell_can_be_expression(self, ev):
        node = MatrixNode(rows=[
            [BinOpNode(NumberNode(1.0), "PLUS", NumberNode(2.0))]
        ])
        result = ev.eval(node)
        assert result[0, 0] == pytest.approx(3.0)

    def test_inconsistent_row_lengths_raise_type_error(self, ev):
        node = MatrixNode(rows=[
            [NumberNode(1.0), NumberNode(2.0)],
            [NumberNode(3.0)],
        ])
        with pytest.raises(TypeError):
            ev.eval(node)

    def test_matrix_cell_non_scalar_raises_type_error(self, ev, st):
        st.set("M", np.array([[1.0, 2.0], [3.0, 4.0]]))
        node = MatrixNode(rows=[[VarNode("M")]])
        with pytest.raises(TypeError):
            ev.eval(node)


# ---------------------------------------------------------------------------
# BinOpNode — PLUS
# ---------------------------------------------------------------------------

class TestEvalBinOpPlus:
    def test_1x1_addition(self, ev):
        node = BinOpNode(NumberNode(2.0), "PLUS", NumberNode(3.0))
        result = ev.eval(node)
        assert result.item() == pytest.approx(5.0)

    def test_2x2_matrix_addition(self, ev):
        result = eval_source("[1, 2; 3, 4] + [5, 6; 7, 8]", ev)
        expected = np.array([[6.0, 8.0], [10.0, 12.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_1x3_matrix_addition(self, ev):
        result = eval_source("[1, 2, 3] + [4, 5, 6]", ev)
        np.testing.assert_array_almost_equal(result, np.array([[5.0, 7.0, 9.0]]))

    def test_addition_shape_mismatch_raises_type_error(self, ev):
        node = BinOpNode(
            MatrixNode(rows=[[NumberNode(1.0), NumberNode(2.0)],
                             [NumberNode(3.0), NumberNode(4.0)]]),
            "PLUS",
            MatrixNode(rows=[[NumberNode(1.0), NumberNode(2.0), NumberNode(3.0)]]),
        )
        with pytest.raises(TypeError):
            ev.eval(node)

    def test_addition_result_is_ndarray(self, ev):
        node = BinOpNode(NumberNode(1.0), "PLUS", NumberNode(1.0))
        assert isinstance(ev.eval(node), np.ndarray)


# ---------------------------------------------------------------------------
# BinOpNode — MINUS
# ---------------------------------------------------------------------------

class TestEvalBinOpMinus:
    def test_1x1_subtraction(self, ev):
        node = BinOpNode(NumberNode(5.0), "MINUS", NumberNode(3.0))
        assert ev.eval(node).item() == pytest.approx(2.0)

    def test_2x2_matrix_subtraction(self, ev):
        result = eval_source("[5, 6; 7, 8] - [1, 2; 3, 4]", ev)
        np.testing.assert_array_almost_equal(result, np.array([[4.0, 4.0], [4.0, 4.0]]))

    def test_subtraction_shape_mismatch_raises_type_error(self, ev):
        node = BinOpNode(
            MatrixNode(rows=[[NumberNode(1.0), NumberNode(2.0)],
                             [NumberNode(3.0), NumberNode(4.0)]]),
            "MINUS",
            MatrixNode(rows=[[NumberNode(1.0), NumberNode(2.0), NumberNode(3.0)]]),
        )
        with pytest.raises(TypeError):
            ev.eval(node)


# ---------------------------------------------------------------------------
# BinOpNode — STAR (matrix multiplication)
# ---------------------------------------------------------------------------

class TestEvalBinOpStar:
    def test_1x1_times_1x1(self, ev):
        node = BinOpNode(NumberNode(3.0), "STAR", NumberNode(4.0))
        assert ev.eval(node).item() == pytest.approx(12.0)

    def test_1x2_times_2x1_gives_1x1(self, ev):
        # [[1, 2]] @ [[3], [4]] = [[11]]
        result = eval_source("[1, 2] * [3; 4]", ev)
        assert result.shape == (1, 1)
        assert result.item() == pytest.approx(11.0)

    def test_2x2_times_2x2(self, ev):
        result = eval_source("[1, 2; 3, 4] * [5, 6; 7, 8]", ev)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_2x3_times_3x2_gives_2x2(self, ev):
        result = eval_source("[1, 2, 3; 4, 5, 6] * [7, 8; 9, 10; 11, 12]", ev)
        expected = np.array([[58.0, 64.0], [139.0, 154.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_inner_dimension_mismatch_raises_type_error(self, ev):
        # 2x3 @ 2x3 is invalid
        node = BinOpNode(
            MatrixNode(rows=[
                [NumberNode(1.0), NumberNode(2.0), NumberNode(3.0)],
                [NumberNode(4.0), NumberNode(5.0), NumberNode(6.0)],
            ]),
            "STAR",
            MatrixNode(rows=[
                [NumberNode(1.0), NumberNode(2.0), NumberNode(3.0)],
                [NumberNode(4.0), NumberNode(5.0), NumberNode(6.0)],
            ]),
        )
        with pytest.raises(TypeError):
            ev.eval(node)

    def test_result_shape_2x3_times_3x2(self, ev):
        result = eval_source("[1, 2, 3; 4, 5, 6] * [7, 8; 9, 10; 11, 12]", ev)
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# UnaryNode — MINUS
# ---------------------------------------------------------------------------

class TestEvalUnaryMinus:
    def test_negate_number(self, ev):
        node = UnaryNode("MINUS", NumberNode(5.0))
        result = ev.eval(node)
        assert result.item() == pytest.approx(-5.0)

    def test_negate_matrix(self, ev):
        node = UnaryNode(
            "MINUS",
            MatrixNode(rows=[[NumberNode(1.0), NumberNode(2.0)]])
        )
        result = ev.eval(node)
        np.testing.assert_array_almost_equal(result, np.array([[-1.0, -2.0]]))

    def test_double_negate_returns_original(self, ev):
        inner = NumberNode(3.0)
        node = UnaryNode("MINUS", UnaryNode("MINUS", inner))
        result = ev.eval(node)
        assert result.item() == pytest.approx(3.0)

    def test_negate_variable(self, ev, st):
        st.set("A", np.array([[4.0]]))
        node = UnaryNode("MINUS", VarNode("A"))
        result = ev.eval(node)
        assert result.item() == pytest.approx(-4.0)


# ---------------------------------------------------------------------------
# AssignmentNode
# ---------------------------------------------------------------------------

class TestEvalAssignmentNode:
    def test_assignment_returns_value(self, ev):
        node = AssignmentNode("A", NumberNode(3.0))
        result = ev.eval(node)
        assert result.item() == pytest.approx(3.0)

    def test_assignment_stores_in_symbol_table(self, ev, st):
        ev.eval(AssignmentNode("A", NumberNode(7.0)))
        assert st.contains("A")
        assert st.get("A").item() == pytest.approx(7.0)

    def test_assignment_stores_matrix(self, ev, st):
        node = AssignmentNode("M", MatrixNode(rows=[
            [NumberNode(1.0), NumberNode(2.0)],
            [NumberNode(3.0), NumberNode(4.0)],
        ]))
        ev.eval(node)
        np.testing.assert_array_almost_equal(
            st.get("M"), np.array([[1.0, 2.0], [3.0, 4.0]])
        )

    def test_assignment_overwrites_previous_value(self, ev, st):
        ev.eval(AssignmentNode("X", NumberNode(1.0)))
        ev.eval(AssignmentNode("X", NumberNode(2.0)))
        assert st.get("X").item() == pytest.approx(2.0)

    def test_assignment_result_is_ndarray(self, ev):
        result = ev.eval(AssignmentNode("A", NumberNode(5.0)))
        assert isinstance(result, np.ndarray)

    def test_chained_usage_after_assignment(self, ev, st):
        ev.eval(AssignmentNode("A", MatrixNode(rows=[
            [NumberNode(1.0), NumberNode(0.0)],
            [NumberNode(0.0), NumberNode(1.0)],
        ])))
        result = ev.eval(VarNode("A"))
        np.testing.assert_array_almost_equal(result, np.eye(2))


# ---------------------------------------------------------------------------
# Built-in: transpose
# ---------------------------------------------------------------------------

class TestBuiltinTranspose:
    def test_transpose_2x2(self, ev, st):
        st.set("A", np.array([[1.0, 2.0], [3.0, 4.0]]))
        result = eval_source("transpose(A)", ev)
        np.testing.assert_array_almost_equal(
            result, np.array([[1.0, 3.0], [2.0, 4.0]])
        )

    def test_transpose_shape_swapped(self, ev, st):
        st.set("A", np.array([[1.0, 2.0, 3.0]]))
        result = eval_source("transpose(A)", ev)
        assert result.shape == (3, 1)

    def test_transpose_of_column_is_row(self, ev, st):
        col = np.array([[1.0], [2.0], [3.0]])
        st.set("v", col)
        result = eval_source("transpose(v)", ev)
        assert result.shape == (1, 3)

    def test_transpose_square_correct_values(self, ev):
        result = eval_source("transpose([1, 2; 3, 4])", ev)
        np.testing.assert_array_almost_equal(
            result, np.array([[1.0, 3.0], [2.0, 4.0]])
        )

    def test_transpose_too_many_args_raises_type_error(self, ev, st):
        st.set("A", np.eye(2))
        with pytest.raises(TypeError):
            eval_source("transpose(A, A)", ev)


# ---------------------------------------------------------------------------
# Built-in: det
# ---------------------------------------------------------------------------

class TestBuiltinDet:
    def test_det_2x2_correct_value(self, ev):
        # det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        result = eval_source("det([1, 2; 3, 4])", ev)
        assert result.item() == pytest.approx(-2.0)

    def test_det_returns_1x1_array(self, ev):
        result = eval_source("det([1, 2; 3, 4])", ev)
        assert result.shape == (1, 1)

    def test_det_identity_2x2_is_1(self, ev):
        result = eval_source("det([1, 0; 0, 1])", ev)
        assert result.item() == pytest.approx(1.0)

    def test_det_singular_matrix_is_zero(self, ev):
        # Rows are proportional → det = 0
        result = eval_source("det([1, 2; 2, 4])", ev)
        assert result.item() == pytest.approx(0.0)

    def test_det_3x3(self, ev):
        result = eval_source("det([6, 1, 1; 4, -2, 5; 2, 8, 7])", ev)
        assert result.item() == pytest.approx(-306.0)

    def test_det_non_square_raises_value_error(self, ev):
        with pytest.raises(ValueError):
            eval_source("det([1, 2, 3; 4, 5, 6])", ev)

    def test_det_1x3_raises_value_error(self, ev):
        with pytest.raises(ValueError):
            eval_source("det([1, 2, 3])", ev)

    def test_det_too_many_args_raises_type_error(self, ev, st):
        st.set("A", np.eye(2))
        with pytest.raises(TypeError):
            eval_source("det(A, A)", ev)


# ---------------------------------------------------------------------------
# Built-in: inv
# ---------------------------------------------------------------------------

class TestBuiltinInv:
    def test_inv_2x2_correct_values(self, ev):
        result = eval_source("inv([1, 2; 3, 4])", ev)
        expected = np.linalg.inv(np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_array_almost_equal(result, expected)

    def test_inv_shape_unchanged(self, ev):
        result = eval_source("inv([1, 2; 3, 4])", ev)
        assert result.shape == (2, 2)

    def test_inv_of_identity_is_identity(self, ev):
        result = eval_source("inv([1, 0; 0, 1])", ev)
        np.testing.assert_array_almost_equal(result, np.eye(2))

    def test_inv_non_square_raises_value_error(self, ev):
        with pytest.raises(ValueError):
            eval_source("inv([1, 2, 3; 4, 5, 6])", ev)

    def test_inv_1x3_raises_value_error(self, ev):
        with pytest.raises(ValueError):
            eval_source("inv([1, 2, 3])", ev)

    def test_inv_too_many_args_raises_type_error(self, ev, st):
        st.set("A", np.eye(2))
        with pytest.raises(TypeError):
            eval_source("inv(A, A)", ev)

    def test_inv_times_original_is_identity(self, ev, st):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        st.set("A", A)
        inv_result = eval_source("inv(A)", ev)
        product = inv_result @ A
        np.testing.assert_array_almost_equal(product, np.eye(2))


# ---------------------------------------------------------------------------
# Built-in: trace
# ---------------------------------------------------------------------------

class TestBuiltinTrace:
    def test_trace_2x2(self, ev):
        # trace([[1,2],[3,4]]) = 1 + 4 = 5
        result = eval_source("trace([1, 2; 3, 4])", ev)
        assert result.item() == pytest.approx(5.0)

    def test_trace_returns_1x1_array(self, ev):
        result = eval_source("trace([1, 2; 3, 4])", ev)
        assert result.shape == (1, 1)

    def test_trace_identity_3x3_is_3(self, ev):
        result = eval_source("trace([1, 0, 0; 0, 1, 0; 0, 0, 1])", ev)
        assert result.item() == pytest.approx(3.0)

    def test_trace_1x1(self, ev):
        result = eval_source("trace([7])", ev)
        assert result.item() == pytest.approx(7.0)

    def test_trace_non_square_uses_diagonal(self, ev):
        # numpy.trace on a 2x3 takes the main diagonal elements
        result = eval_source("trace([1, 2, 3; 4, 5, 6])", ev)
        assert result.item() == pytest.approx(6.0)  # 1 + 5

    def test_trace_too_many_args_raises_type_error(self, ev, st):
        st.set("A", np.eye(2))
        with pytest.raises(TypeError):
            eval_source("trace(A, A)", ev)


# ---------------------------------------------------------------------------
# Built-in: eye
# ---------------------------------------------------------------------------

class TestBuiltinEye:
    def test_eye_3_shape(self, ev):
        result = eval_source("eye(3)", ev)
        assert result.shape == (3, 3)

    def test_eye_3_values(self, ev):
        result = eval_source("eye(3)", ev)
        np.testing.assert_array_almost_equal(result, np.eye(3))

    def test_eye_1_is_1x1_identity(self, ev):
        result = eval_source("eye(1)", ev)
        np.testing.assert_array_almost_equal(result, np.array([[1.0]]))

    def test_eye_2_diagonal_is_one(self, ev):
        result = eval_source("eye(2)", ev)
        assert result[0, 0] == pytest.approx(1.0)
        assert result[1, 1] == pytest.approx(1.0)

    def test_eye_2_off_diagonal_is_zero(self, ev):
        result = eval_source("eye(2)", ev)
        assert result[0, 1] == pytest.approx(0.0)
        assert result[1, 0] == pytest.approx(0.0)

    def test_eye_zero_raises_value_error(self, ev):
        with pytest.raises(ValueError):
            eval_source("eye(0)", ev)

    def test_eye_negative_raises_value_error(self, ev):
        with pytest.raises(ValueError):
            eval_source("eye(-1)", ev)

    def test_eye_too_many_args_raises_type_error(self, ev):
        with pytest.raises(TypeError):
            eval_source("eye(2, 3)", ev)


# ---------------------------------------------------------------------------
# Built-in: zeros
# ---------------------------------------------------------------------------

class TestBuiltinZeros:
    def test_zeros_2x3_shape(self, ev):
        result = eval_source("zeros(2, 3)", ev)
        assert result.shape == (2, 3)

    def test_zeros_2x3_all_zero(self, ev):
        result = eval_source("zeros(2, 3)", ev)
        np.testing.assert_array_almost_equal(result, np.zeros((2, 3)))

    def test_zeros_1x1(self, ev):
        result = eval_source("zeros(1, 1)", ev)
        assert result.item() == pytest.approx(0.0)

    def test_zeros_4x4_shape(self, ev):
        result = eval_source("zeros(4, 4)", ev)
        assert result.shape == (4, 4)

    def test_zeros_zero_rows_raises_value_error(self, ev):
        with pytest.raises(ValueError):
            eval_source("zeros(0, 3)", ev)

    def test_zeros_zero_cols_raises_value_error(self, ev):
        with pytest.raises(ValueError):
            eval_source("zeros(2, 0)", ev)

    def test_zeros_one_arg_raises_type_error(self, ev):
        with pytest.raises(TypeError):
            eval_source("zeros(2)", ev)


# ---------------------------------------------------------------------------
# Built-in: ones
# ---------------------------------------------------------------------------

class TestBuiltinOnes:
    def test_ones_2x3_shape(self, ev):
        result = eval_source("ones(2, 3)", ev)
        assert result.shape == (2, 3)

    def test_ones_2x3_all_one(self, ev):
        result = eval_source("ones(2, 3)", ev)
        np.testing.assert_array_almost_equal(result, np.ones((2, 3)))

    def test_ones_1x1(self, ev):
        result = eval_source("ones(1, 1)", ev)
        assert result.item() == pytest.approx(1.0)

    def test_ones_4x4_shape(self, ev):
        result = eval_source("ones(4, 4)", ev)
        assert result.shape == (4, 4)

    def test_ones_zero_rows_raises_value_error(self, ev):
        with pytest.raises(ValueError):
            eval_source("ones(0, 3)", ev)

    def test_ones_zero_cols_raises_value_error(self, ev):
        with pytest.raises(ValueError):
            eval_source("ones(2, 0)", ev)

    def test_ones_one_arg_raises_type_error(self, ev):
        with pytest.raises(TypeError):
            eval_source("ones(3)", ev)


# ---------------------------------------------------------------------------
# Built-in: dagger (conjugate transpose)
# ---------------------------------------------------------------------------

class TestBuiltinDagger:
    def test_dagger_real_matrix_is_transpose(self, ev):
        result = eval_source("dagger([1, 2; 3, 4])", ev)
        np.testing.assert_array_almost_equal(
            result, np.array([[1.0, 3.0], [2.0, 4.0]])
        )

    def test_dagger_shape_swapped(self, ev, st):
        st.set("A", np.array([[1.0, 2.0, 3.0]]))
        result = eval_source("dagger(A)", ev)
        assert result.shape == (3, 1)

    def test_dagger_conjugates_complex_matrix(self, ev, st):
        A = np.array([[1 + 2j, 3 - 1j], [0 + 1j, 2 + 0j]])
        st.set("A", A)
        result = eval_source("dagger(A)", ev)
        np.testing.assert_array_almost_equal(result, A.conj().T)

    def test_dagger_twice_returns_original(self, ev, st):
        A = np.array([[1 + 2j, 3 - 1j], [0 + 1j, 2 + 0j]])
        st.set("A", A)
        result = eval_source("dagger(dagger(A))", ev)
        np.testing.assert_array_almost_equal(result, A)

    def test_dagger_too_many_args_raises_type_error(self, ev, st):
        st.set("A", np.eye(2))
        with pytest.raises(TypeError):
            eval_source("dagger(A, A)", ev)


# ---------------------------------------------------------------------------
# Built-in: outer (outer product)
# ---------------------------------------------------------------------------

class TestBuiltinOuter:
    def test_outer_row_vectors(self, ev):
        result = eval_source("outer([1, 2], [3, 4])", ev)
        expected = np.outer([1.0, 2.0], [3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_outer_shape(self, ev):
        result = eval_source("outer([1, 2, 3], [4, 5])", ev)
        assert result.shape == (3, 2)

    def test_outer_column_vectors(self, ev):
        result = eval_source("outer([1; 2], [3; 4])", ev)
        expected = np.outer([1.0, 2.0], [3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_outer_mixed_row_and_column(self, ev):
        result = eval_source("outer([1, 2], [3; 4])", ev)
        expected = np.outer([1.0, 2.0], [3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_outer_of_qubit_basis_states(self, ev):
        # |0><1| for a single qubit: outer([1,0], [0,1])
        result = eval_source("outer([1, 0], [0, 1])", ev)
        np.testing.assert_array_almost_equal(result, np.array([[0.0, 1.0], [0.0, 0.0]]))

    def test_outer_non_vector_raises_type_error(self, ev):
        with pytest.raises(TypeError):
            eval_source("outer([1, 2; 3, 4], [1, 2])", ev)

    def test_outer_wrong_arg_count_raises_type_error(self, ev):
        with pytest.raises(TypeError):
            eval_source("outer([1, 2])", ev)


# ---------------------------------------------------------------------------
# Built-in: tensor / kron (tensor product)
# ---------------------------------------------------------------------------

class TestBuiltinTensor:
    def test_tensor_2x2_identity_with_itself(self, ev):
        result = eval_source("tensor([1, 0; 0, 1], [1, 0; 0, 1])", ev)
        np.testing.assert_array_almost_equal(result, np.eye(4))

    def test_tensor_shape(self, ev):
        result = eval_source("tensor([1, 2], [1, 0; 0, 1])", ev)
        assert result.shape == (2, 4)

    def test_tensor_matches_numpy_kron(self, ev, st):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[0.0, 1.0], [1.0, 0.0]])
        st.set("A", A)
        st.set("B", B)
        result = eval_source("tensor(A, B)", ev)
        np.testing.assert_array_almost_equal(result, np.kron(A, B))

    def test_tensor_of_two_qubits(self, ev):
        # |0> tensor |1> = [0, 1, 0, 0]
        result = eval_source("tensor([1; 0], [0; 1])", ev)
        np.testing.assert_array_almost_equal(result, np.array([[0.0], [1.0], [0.0], [0.0]]))

    def test_tensor_too_few_args_raises_type_error(self, ev, st):
        st.set("A", np.eye(2))
        with pytest.raises(TypeError):
            eval_source("tensor(A)", ev)

    def test_kron_is_alias_for_tensor(self, ev, st):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[0.0, 1.0], [1.0, 0.0]])
        st.set("A", A)
        st.set("B", B)
        tensor_result = eval_source("tensor(A, B)", ev)
        kron_result = eval_source("kron(A, B)", ev)
        np.testing.assert_array_almost_equal(tensor_result, kron_result)

    def test_kron_matches_numpy_kron(self, ev, st):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[0.0, 1.0], [1.0, 0.0]])
        st.set("A", A)
        st.set("B", B)
        result = eval_source("kron(A, B)", ev)
        np.testing.assert_array_almost_equal(result, np.kron(A, B))


# ---------------------------------------------------------------------------
# Built-in: commutator
# ---------------------------------------------------------------------------

class TestBuiltinCommutator:
    def test_commutator_of_identity_is_zero(self, ev):
        result = eval_source("commutator([1, 0; 0, 1], [1, 2; 3, 4])", ev)
        np.testing.assert_array_almost_equal(result, np.zeros((2, 2)))

    def test_commutator_matches_manual_computation(self, ev, st):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[0.0, 1.0], [1.0, 0.0]])
        st.set("A", A)
        st.set("B", B)
        result = eval_source("commutator(A, B)", ev)
        np.testing.assert_array_almost_equal(result, A @ B - B @ A)

    def test_commutator_pauli_x_pauli_y(self, ev):
        # [X, Y] = 2iZ; using real Pauli-like matrices without the
        # imaginary unit still exercises the anti-commuting structure.
        result = eval_source(
            "commutator([0, 1; 1, 0], [0, -1; 1, 0])", ev
        )
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        Y = np.array([[0.0, -1.0], [1.0, 0.0]])
        np.testing.assert_array_almost_equal(result, X @ Y - Y @ X)

    def test_commutator_antisymmetric(self, ev, st):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[0.0, 1.0], [1.0, 0.0]])
        st.set("A", A)
        st.set("B", B)
        ab = eval_source("commutator(A, B)", ev)
        ba = eval_source("commutator(B, A)", ev)
        np.testing.assert_array_almost_equal(ab, -ba)

    def test_commutator_non_square_raises_value_error(self, ev):
        with pytest.raises(ValueError):
            eval_source("commutator([1, 2, 3; 4, 5, 6], [1, 0; 0, 1])", ev)

    def test_commutator_mismatched_shapes_raises_value_error(self, ev):
        with pytest.raises(ValueError):
            eval_source("commutator([1, 0; 0, 1], [1, 0, 0; 0, 1, 0; 0, 0, 1])", ev)

    def test_commutator_wrong_arg_count_raises_type_error(self, ev, st):
        st.set("A", np.eye(2))
        with pytest.raises(TypeError):
            eval_source("commutator(A)", ev)


# ---------------------------------------------------------------------------
# Unknown function name
# ---------------------------------------------------------------------------

class TestUnknownFunction:
    def test_unknown_function_raises_name_error(self, ev):
        node = FunctionNode("foobar", [NumberNode(1.0)])
        with pytest.raises(NameError):
            ev.eval(node)

    def test_name_error_message_contains_function_name(self, ev):
        node = FunctionNode("foobar", [NumberNode(1.0)])
        with pytest.raises(NameError, match="foobar"):
            ev.eval(node)


# ---------------------------------------------------------------------------
# Integration via eval_source — end-to-end pipeline
# ---------------------------------------------------------------------------

class TestEndToEndEval:
    def test_matrix_addition_row_vector(self, ev):
        result = eval_source("[1, 2] + [3, 4]", ev)
        np.testing.assert_array_almost_equal(result, np.array([[4.0, 6.0]]))

    def test_matrix_multiply_row_by_col(self, ev):
        # [[1,2]] * [[3],[4]] = [[11]]
        result = eval_source("[1, 2] * [3; 4]", ev)
        assert result.item() == pytest.approx(11.0)

    def test_determinant_1_2_3_4(self, ev):
        result = eval_source("det([1, 2; 3, 4])", ev)
        assert result.item() == pytest.approx(-2.0)

    def test_identity_eye_2(self, ev):
        result = eval_source("eye(2)", ev)
        np.testing.assert_array_almost_equal(result, np.array([[1.0, 0.0], [0.0, 1.0]]))

    def test_assignment_then_use(self, ev, st):
        eval_source("A = [1, 2; 3, 4]", ev)
        result = eval_source("det(A)", ev)
        assert result.item() == pytest.approx(-2.0)

    def test_transpose_then_multiply(self, ev, st):
        st.set("A", np.array([[1.0, 2.0], [3.0, 4.0]]))
        result = eval_source("transpose(A) * A", ev)
        expected = np.array([[1.0, 3.0], [2.0, 4.0]]) @ np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_complex_expression_precedence(self, ev):
        # 2 + 3 * 4 should be 2 + 12 = 14
        result = eval_source("2 + 3 * 4", ev)
        assert result.item() == pytest.approx(14.0)

    def test_zeros_plus_ones(self, ev):
        result = eval_source("zeros(2, 2) + ones(2, 2)", ev)
        np.testing.assert_array_almost_equal(result, np.ones((2, 2)))

    def test_inverse_of_eye_is_eye(self, ev):
        result = eval_source("inv(eye(3))", ev)
        np.testing.assert_array_almost_equal(result, np.eye(3))
