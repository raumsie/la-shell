"""Symbol table for the LA Shell interpreter.

Provides a persistent, named variable store that maps identifier strings
to NumPy array values for the duration of a REPL session.
"""

from __future__ import annotations

import numpy as np


class SymbolTable:
    """Persistent variable store for the LA Shell session.

    Stores named bindings from identifier strings to NumPy arrays.
    All mutation is explicit — there is no implicit scoping or shadowing.

    Example:
        >>> st = SymbolTable()
        >>> st.set("A", np.array([[1.0, 2.0], [3.0, 4.0]]))
        >>> st.contains("A")
        True
        >>> st.get("A")
        array([[1., 2.],
               [3., 4.]])
    """

    def __init__(self) -> None:
        """Initialise an empty symbol table."""
        self._vars: dict[str, np.ndarray] = {}

    def set(self, name: str, value: np.ndarray) -> None:
        """Bind *name* to *value*, overwriting any previous binding.

        Args:
            name: The variable identifier (e.g. ``"A"``).
            value: The NumPy array to associate with *name*.
        """
        self._vars[name] = value

    def get(self, name: str) -> np.ndarray:
        """Return the value currently bound to *name*.

        Args:
            name: The variable identifier to look up.

        Returns:
            The NumPy array stored under *name*.

        Raises:
            NameError: If *name* has not been defined in this session.
        """
        if name not in self._vars:
            raise NameError(f"Variable '{name}' not defined")
        return self._vars[name]

    def contains(self, name: str) -> bool:
        """Return ``True`` if *name* is currently defined, else ``False``.

        Args:
            name: The variable identifier to check.

        Returns:
            Boolean indicating presence of *name* in the table.
        """
        return name in self._vars

    def list_all(self) -> dict[str, np.ndarray]:
        """Return a shallow copy of all current variable bindings.

        Returns:
            A dictionary mapping each defined identifier to its value.
            Mutations to the returned dict do not affect the table.
        """
        return self._vars.copy()

    def clear(self) -> None:
        """Remove all variable bindings, resetting the table to empty."""
        self._vars.clear()
