"""AST evaluator for the LA Shell interpreter.

Walks a parsed AST produced by ``src.parser`` and executes each node
using NumPy for all numerical and matrix operations.

All scalar results are returned as 1×1 NumPy arrays so that the rest of
the pipeline can always treat a result as ``np.ndarray``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.parser import (
    AssignmentNode,
    BinOpNode,
    FunctionNode,
    MatrixNode,
    NumberNode,
    UnaryNode,
    VarNode,
)
from src.symbol_table import SymbolTable

if TYPE_CHECKING:
    # Union of every concrete AST node type.
    from src.parser import (
        AssignmentNode,
        BinOpNode,
        FunctionNode,
        MatrixNode,
        NumberNode,
        UnaryNode,
        VarNode,
    )

# Type alias used in docstrings only.
_ASTNode = object


class Evaluator:
    """Walks an AST and computes a NumPy result for each node.

    The evaluator is stateful: assignments update the ``SymbolTable``
    supplied at construction time, and variable references read from it.

    Args:
        symbol_table: The persistent variable store for this REPL session.

    Example:
        >>> from src.symbol_table import SymbolTable
        >>> st = SymbolTable()
        >>> ev = Evaluator(st)
        >>> # Assume `ast` is a parsed AssignmentNode for  A = [[1,2];[3,4]]
        >>> result = ev.eval(ast)
    """

    def __init__(self, symbol_table: SymbolTable) -> None:
        """Initialise the evaluator with a shared symbol table.

        Args:
            symbol_table: Mutable store shared with the REPL and commands.
        """
        self.symbol_table: SymbolTable = symbol_table

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    def eval(self, node: _ASTNode) -> np.ndarray:
        """Recursively evaluate *node* and return a NumPy array result.

        Args:
            node: Any concrete AST node returned by ``Parser.parse()``.

        Returns:
            A NumPy ``ndarray``.  Scalars are returned as 1×1 arrays.

        Raises:
            NameError: If a referenced variable or function is not defined.
            TypeError: If operand shapes are incompatible for the requested
                operation (e.g. dimension mismatch in matrix multiplication
                or addition).
            ValueError: If a function receives an argument with an invalid
                shape (e.g. ``det`` on a non-square matrix).
            RuntimeError: If an unknown AST node type is encountered
                (indicates a parser/evaluator version mismatch).
        """
        if isinstance(node, NumberNode):
            return self._eval_number(node)

        if isinstance(node, VarNode):
            return self._eval_var(node)

        if isinstance(node, MatrixNode):
            return self._eval_matrix(node)

        if isinstance(node, BinOpNode):
            return self._eval_binop(node)

        if isinstance(node, UnaryNode):
            return self._eval_unary(node)

        if isinstance(node, FunctionNode):
            return self._eval_function_node(node)

        if isinstance(node, AssignmentNode):
            return self._eval_assignment(node)

        raise RuntimeError(
            f"Unknown AST node type: {type(node).__name__}. "
            "Ensure the parser and evaluator versions match."
        )

    # ------------------------------------------------------------------
    # Node handlers
    # ------------------------------------------------------------------

    def _eval_number(self, node: NumberNode) -> np.ndarray:
        """Return a numeric literal as a 1×1 array.

        Args:
            node: A ``NumberNode`` carrying a ``float`` value.

        Returns:
            ``np.array([[value]])`` with dtype ``float64``.
        """
        return np.array([[node.value]], dtype=float)

    def _eval_var(self, node: VarNode) -> np.ndarray:
        """Look up a variable in the symbol table.

        Args:
            node: A ``VarNode`` carrying the identifier name.

        Returns:
            The NumPy array currently bound to the variable.

        Raises:
            NameError: If the variable has not been defined yet.
        """
        if not self.symbol_table.contains(node.name):
            raise NameError(f"Variable '{node.name}' not defined")
        return self.symbol_table.get(node.name)

    def _eval_matrix(self, node: MatrixNode) -> np.ndarray:
        """Build a 2-D NumPy array from a ``MatrixNode``.

        Each cell expression inside the matrix literal is evaluated
        recursively.  A cell that evaluates to a 1×1 array is collapsed
        to a plain Python float via ``.item()`` so that ``np.array`` can
        construct a homogeneous 2-D matrix.

        Args:
            node: A ``MatrixNode`` whose ``rows`` attribute is a list of
                lists of AST nodes.

        Returns:
            A 2-D ``float64`` array with shape ``(num_rows, num_cols)``.

        Raises:
            TypeError: If rows have inconsistent column counts.
        """
        evaluated_rows: list[list[float]] = []
        num_cols: int | None = None

        for row_idx, row in enumerate(node.rows):
            row_vals: list[float] = []
            for cell in row:
                cell_result = self.eval(cell)
                # Each cell must reduce to a single scalar.
                if cell_result.size != 1:
                    raise TypeError(
                        f"Matrix literal cell at row {row_idx} evaluated to a "
                        f"{cell_result.shape[0]}×{cell_result.shape[1]} matrix; "
                        "each cell must be a scalar expression."
                    )
                row_vals.append(cell_result.item())

            if num_cols is None:
                num_cols = len(row_vals)
            elif len(row_vals) != num_cols:
                raise TypeError(
                    f"Inconsistent row lengths in matrix literal: "
                    f"expected {num_cols} columns but row {row_idx} has "
                    f"{len(row_vals)} columns."
                )

            evaluated_rows.append(row_vals)

        return np.array(evaluated_rows, dtype=float)

    def _eval_binop(self, node: BinOpNode) -> np.ndarray:
        """Evaluate a binary operation node.

        Supported operators:

        * ``PLUS``  — element-wise addition (shapes must match).
        * ``MINUS`` — element-wise subtraction (shapes must match).
        * ``STAR``  — matrix multiplication via NumPy ``@`` operator.

        1×1 arrays (scalars) are automatically broadcast by NumPy for
        addition and subtraction.  For multiplication a scalar (1×1) on
        either side scales the other operand element-wise using ``@``,
        which is equivalent to scalar multiplication in that degenerate
        case.

        Args:
            node: A ``BinOpNode`` with ``left``, ``op``, and ``right``.

        Returns:
            The result as a NumPy array.

        Raises:
            TypeError: If the operand shapes are incompatible.
        """
        left = self.eval(node.left)
        right = self.eval(node.right)

        if node.op == "PLUS":
            return self._safe_add(left, right)

        if node.op == "MINUS":
            return self._safe_subtract(left, right)

        if node.op == "STAR":
            return self._safe_matmul(left, right)

        raise RuntimeError(f"Unknown binary operator: '{node.op}'")

    def _eval_unary(self, node: UnaryNode) -> np.ndarray:
        """Evaluate a unary negation node.

        Args:
            node: A ``UnaryNode`` with op ``'MINUS'``.

        Returns:
            Element-wise negation of the operand.
        """
        operand = self.eval(node.operand)
        if node.op == "MINUS":
            return -operand
        raise RuntimeError(f"Unknown unary operator: '{node.op}'")

    def _eval_function_node(self, node: FunctionNode) -> np.ndarray:
        """Dispatch a built-in function call node.

        Args:
            node: A ``FunctionNode`` with a ``name`` and ``args`` list.

        Returns:
            The function result as a NumPy array.
        """
        return self.eval_function(node.name, node.args)

    def _eval_assignment(self, node: AssignmentNode) -> np.ndarray:
        """Evaluate the RHS expression and bind the result to the variable.

        Args:
            node: An ``AssignmentNode`` with a ``name`` and ``expr``.

        Returns:
            The value that was stored (same object the caller can display).
        """
        value = self.eval(node.expr)
        self.symbol_table.set(node.name, value)
        return value

    # ------------------------------------------------------------------
    # Built-in functions
    # ------------------------------------------------------------------

    def eval_function(self, name: str, args: list[_ASTNode]) -> np.ndarray:
        """Evaluate a built-in function by name.

        Evaluates each argument expression first, then applies the
        corresponding NumPy operation.

        Supported functions:

        +------------+------------------------+------------------------------------+
        | Name       | Signature              | Description                        |
        +============+========================+====================================+
        | transpose  | ``transpose(A)``       | Matrix transpose                   |
        +------------+------------------------+------------------------------------+
        | det        | ``det(A)``             | Determinant (square matrices only) |
        +------------+------------------------+------------------------------------+
        | inv        | ``inv(A)``             | Matrix inverse (square only)       |
        +------------+------------------------+------------------------------------+
        | trace      | ``trace(A)``           | Sum of diagonal elements           |
        +------------+------------------------+------------------------------------+
        | eye        | ``eye(n)``             | n×n identity matrix                |
        +------------+------------------------+------------------------------------+
        | zeros      | ``zeros(r, c)``        | r×c zero matrix                    |
        +------------+------------------------+------------------------------------+
        | ones       | ``ones(r, c)``         | r×c ones matrix                    |
        +------------+------------------------+------------------------------------+
        | dagger     | ``dagger(A)``          | Conjugate transpose (A^dagger)     |
        +------------+------------------------+------------------------------------+
        | outer      | ``outer(u, v)``        | Outer product of two vectors       |
        +------------+------------------------+------------------------------------+
        | tensor     | ``tensor(A, B)``       | Tensor / Kronecker product         |
        +------------+------------------------+------------------------------------+
        | kron       | ``kron(A, B)``         | Alias for ``tensor(A, B)``         |
        +------------+------------------------+------------------------------------+
        | commutator | ``commutator(A, B)``   | ``A*B - B*A`` (square, same shape) |
        +------------+------------------------+------------------------------------+

        Args:
            name: The function name as it appears in the source.
            args: Unevaluated AST argument nodes.

        Returns:
            A NumPy array result.  Scalar outputs (det, trace) are
            returned as 1×1 arrays.

        Raises:
            NameError: If *name* is not a recognised built-in.
            TypeError: If the wrong number of arguments is supplied, or if
                ``outer()`` is given a non-vector argument.
            ValueError: If a matrix argument has an invalid shape for the
                requested operation (e.g. non-square for ``det`` or ``inv``,
                or mismatched/non-square shapes for ``commutator``).
        """
        evaluated = [self.eval(arg) for arg in args]

        if name == "transpose":
            self._check_arg_count(name, evaluated, expected=1)
            return evaluated[0].T

        if name == "det":
            self._check_arg_count(name, evaluated, expected=1)
            matrix = evaluated[0]
            self._require_square(matrix, "det")
            return np.array([[np.linalg.det(matrix)]])

        if name == "inv":
            self._check_arg_count(name, evaluated, expected=1)
            matrix = evaluated[0]
            self._require_square(matrix, "inv")
            try:
                return np.linalg.inv(matrix)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    f"Cannot invert matrix: {exc}"
                ) from exc

        if name == "trace":
            self._check_arg_count(name, evaluated, expected=1)
            return np.array([[np.trace(evaluated[0])]])

        if name == "eye":
            self._check_arg_count(name, evaluated, expected=1)
            n = self._scalar_int(evaluated[0], "eye", "n")
            if n < 1:
                raise ValueError(
                    f"eye() requires a positive integer; got {n}."
                )
            return np.eye(n)

        if name == "zeros":
            self._check_arg_count(name, evaluated, expected=2)
            r = self._scalar_int(evaluated[0], "zeros", "r")
            c = self._scalar_int(evaluated[1], "zeros", "c")
            if r < 1 or c < 1:
                raise ValueError(
                    f"zeros() requires positive dimensions; got ({r}, {c})."
                )
            return np.zeros((r, c))

        if name == "ones":
            self._check_arg_count(name, evaluated, expected=2)
            r = self._scalar_int(evaluated[0], "ones", "r")
            c = self._scalar_int(evaluated[1], "ones", "c")
            if r < 1 or c < 1:
                raise ValueError(
                    f"ones() requires positive dimensions; got ({r}, {c})."
                )
            return np.ones((r, c))

        if name == "dagger":
            self._check_arg_count(name, evaluated, expected=1)
            return np.conjugate(evaluated[0]).T

        if name == "outer":
            self._check_arg_count(name, evaluated, expected=2)
            u = self._as_vector(evaluated[0], "outer", "u")
            v = self._as_vector(evaluated[1], "outer", "v")
            return np.outer(u, v)

        # tensor(A, B) and its alias kron(A, B) both compute the Kronecker
        # (tensor) product; kron is accepted silently for users coming from
        # NumPy/MATLAB habits.
        if name in ("tensor", "kron"):
            self._check_arg_count(name, evaluated, expected=2)
            return np.kron(evaluated[0], evaluated[1])

        if name == "commutator":
            self._check_arg_count(name, evaluated, expected=2)
            a, b = evaluated[0], evaluated[1]
            self._require_square(a, "commutator")
            self._require_square(b, "commutator")
            if a.shape != b.shape:
                raise ValueError(
                    f"commutator() requires matrices of the same shape; "
                    f"got {a.shape[0]}×{a.shape[1]} and {b.shape[0]}×{b.shape[1]}."
                )
            return a @ b - b @ a

        raise NameError(f"Unknown function: '{name}'")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Element-wise addition with a descriptive error on shape mismatch.

        Args:
            left: Left operand.
            right: Right operand.

        Returns:
            ``left + right``

        Raises:
            TypeError: If shapes are incompatible for broadcasting.
        """
        try:
            return left + right
        except ValueError as exc:
            raise TypeError(
                f"Cannot add {left.shape[0]}×{left.shape[1]} matrix to "
                f"{right.shape[0]}×{right.shape[1]} matrix: shapes do not match."
            ) from exc

    @staticmethod
    def _safe_subtract(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Element-wise subtraction with a descriptive error on shape mismatch.

        Args:
            left: Left operand.
            right: Right operand.

        Returns:
            ``left - right``

        Raises:
            TypeError: If shapes are incompatible for broadcasting.
        """
        try:
            return left - right
        except ValueError as exc:
            raise TypeError(
                f"Cannot subtract {right.shape[0]}×{right.shape[1]} matrix from "
                f"{left.shape[0]}×{left.shape[1]} matrix: shapes do not match."
            ) from exc

    @staticmethod
    def _safe_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Matrix multiplication (``@``) with a descriptive error on mismatch.

        Args:
            left: Left operand.
            right: Right operand.

        Returns:
            ``left @ right``

        Raises:
            TypeError: If the inner dimensions do not match.
        """
        try:
            return left @ right
        except ValueError as exc:
            raise TypeError(
                f"Cannot multiply {left.shape[0]}×{left.shape[1]} matrix by "
                f"{right.shape[0]}×{right.shape[1]} matrix: "
                f"inner dimensions {left.shape[1]} and {right.shape[0]} do not match."
            ) from exc

    @staticmethod
    def _require_square(matrix: np.ndarray, func_name: str) -> None:
        """Raise ``ValueError`` if *matrix* is not square.

        Args:
            matrix: The matrix to check.
            func_name: The calling function's name (used in the error message).

        Raises:
            ValueError: If the matrix is not 2-D or not square.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            rows, cols = matrix.shape[0], matrix.shape[1]
            raise ValueError(
                f"Matrix must be square for {func_name}(); "
                f"got {rows}×{cols} matrix."
            )

    @staticmethod
    def _check_arg_count(
        func_name: str, evaluated: list[np.ndarray], expected: int
    ) -> None:
        """Raise ``TypeError`` if the argument count does not match.

        Args:
            func_name: Name of the function being called.
            evaluated: Already-evaluated argument list.
            expected: Required number of arguments.

        Raises:
            TypeError: If ``len(evaluated) != expected``.
        """
        if len(evaluated) != expected:
            raise TypeError(
                f"{func_name}() takes {expected} argument(s), "
                f"but {len(evaluated)} were given."
            )

    @staticmethod
    def _as_vector(array: np.ndarray, func_name: str, param_name: str) -> np.ndarray:
        """Flatten a row or column vector to 1-D, rejecting other shapes.

        Args:
            array: The evaluated argument; must be a row (1×n) or column
                (n×1) matrix.
            func_name: Name of the calling function (for error messages).
            param_name: Name of the parameter (for error messages).

        Returns:
            A 1-D NumPy array with ``n`` elements.

        Raises:
            TypeError: If *array* is not a row or column vector.
        """
        if array.ndim != 2 or (array.shape[0] != 1 and array.shape[1] != 1):
            raise TypeError(
                f"{func_name}(): argument '{param_name}' must be a row or "
                f"column vector, got a {array.shape[0]}×{array.shape[1]} matrix."
            )
        return array.reshape(-1)

    @staticmethod
    def _scalar_int(array: np.ndarray, func_name: str, param_name: str) -> int:
        """Extract a single integer from a 1×1 array argument.

        Args:
            array: The evaluated argument; must contain exactly one element.
            func_name: Name of the calling function (for error messages).
            param_name: Name of the parameter (for error messages).

        Returns:
            The integer value.

        Raises:
            TypeError: If *array* is not a scalar (1×1) value.
        """
        if array.size != 1:
            raise TypeError(
                f"{func_name}(): argument '{param_name}' must be a scalar, "
                f"got a {array.shape[0]}×{array.shape[1]} matrix."
            )
        return int(array.item())
