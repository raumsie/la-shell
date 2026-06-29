"""Tests for src.symbol_table — SymbolTable."""
from __future__ import annotations

import pytest
import numpy as np

from src.symbol_table import SymbolTable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def table() -> SymbolTable:
    return SymbolTable()


@pytest.fixture
def populated_table() -> SymbolTable:
    st = SymbolTable()
    st.set("A", np.array([[1.0, 2.0], [3.0, 4.0]]))
    st.set("B", np.array([[5.0]]))
    return st


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInitialisation:
    def test_new_table_is_empty(self, table):
        assert table.list_all() == {}

    def test_new_table_contains_nothing(self, table):
        assert not table.contains("A")


# ---------------------------------------------------------------------------
# set and get round-trip
# ---------------------------------------------------------------------------

class TestSetAndGet:
    def test_set_and_get_scalar_array(self, table):
        arr = np.array([[7.0]])
        table.set("x", arr)
        result = table.get("x")
        np.testing.assert_array_equal(result, arr)

    def test_set_and_get_2x2_matrix(self, table):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        table.set("M", arr)
        result = table.get("M")
        np.testing.assert_array_equal(result, arr)

    def test_set_overwrites_previous_value(self, table):
        table.set("A", np.array([[1.0]]))
        table.set("A", np.array([[2.0]]))
        np.testing.assert_array_equal(table.get("A"), np.array([[2.0]]))

    def test_multiple_variables_stored_independently(self, table):
        a = np.array([[1.0]])
        b = np.array([[2.0]])
        table.set("A", a)
        table.set("B", b)
        np.testing.assert_array_equal(table.get("A"), a)
        np.testing.assert_array_equal(table.get("B"), b)

    def test_get_returns_same_object_set(self, table):
        arr = np.array([[99.0]])
        table.set("Z", arr)
        assert table.get("Z") is arr

    def test_variable_name_is_case_sensitive(self, table):
        table.set("A", np.array([[1.0]]))
        table.set("a", np.array([[2.0]]))
        assert not np.array_equal(table.get("A"), table.get("a"))


# ---------------------------------------------------------------------------
# NameError on missing variable
# ---------------------------------------------------------------------------

class TestGetRaisesNameError:
    def test_get_undefined_variable_raises_name_error(self, table):
        with pytest.raises(NameError):
            table.get("X")

    def test_name_error_message_contains_variable_name(self, table):
        with pytest.raises(NameError, match="X"):
            table.get("X")

    def test_get_after_clear_raises_name_error(self, populated_table):
        populated_table.clear()
        with pytest.raises(NameError):
            populated_table.get("A")

    def test_get_wrong_case_raises_name_error(self, table):
        table.set("A", np.array([[1.0]]))
        with pytest.raises(NameError):
            table.get("a")


# ---------------------------------------------------------------------------
# contains
# ---------------------------------------------------------------------------

class TestContains:
    def test_contains_returns_true_for_defined_variable(self, populated_table):
        assert populated_table.contains("A") is True

    def test_contains_returns_false_for_undefined_variable(self, table):
        assert table.contains("Z") is False

    def test_contains_returns_false_after_clear(self, populated_table):
        populated_table.clear()
        assert populated_table.contains("A") is False

    def test_contains_is_case_sensitive(self, table):
        table.set("A", np.array([[1.0]]))
        assert table.contains("A") is True
        assert table.contains("a") is False

    def test_contains_true_for_all_set_variables(self, populated_table):
        assert populated_table.contains("A") is True
        assert populated_table.contains("B") is True


# ---------------------------------------------------------------------------
# list_all
# ---------------------------------------------------------------------------

class TestListAll:
    def test_list_all_returns_all_variable_names(self, populated_table):
        result = populated_table.list_all()
        assert "A" in result
        assert "B" in result

    def test_list_all_returns_correct_values(self, table):
        arr = np.array([[3.0]])
        table.set("X", arr)
        result = table.list_all()
        np.testing.assert_array_equal(result["X"], arr)

    def test_list_all_returns_copy_not_original(self, table):
        table.set("A", np.array([[1.0]]))
        copy = table.list_all()
        copy["NEW_KEY"] = np.array([[99.0]])
        # Mutating returned dict must not affect the table
        assert not table.contains("NEW_KEY")

    def test_list_all_on_empty_table_returns_empty_dict(self, table):
        assert table.list_all() == {}

    def test_list_all_count_matches_set_calls(self, table):
        table.set("A", np.array([[1.0]]))
        table.set("B", np.array([[2.0]]))
        table.set("C", np.array([[3.0]]))
        assert len(table.list_all()) == 3

    def test_list_all_overwrite_does_not_increase_count(self, table):
        table.set("A", np.array([[1.0]]))
        table.set("A", np.array([[2.0]]))
        assert len(table.list_all()) == 1


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_empties_the_table(self, populated_table):
        populated_table.clear()
        assert populated_table.list_all() == {}

    def test_clear_makes_all_variables_undefined(self, populated_table):
        populated_table.clear()
        assert not populated_table.contains("A")
        assert not populated_table.contains("B")

    def test_clear_on_empty_table_is_idempotent(self, table):
        table.clear()
        assert table.list_all() == {}

    def test_set_after_clear_works(self, populated_table):
        populated_table.clear()
        populated_table.set("New", np.array([[1.0]]))
        assert populated_table.contains("New")

    def test_clear_twice_is_safe(self, populated_table):
        populated_table.clear()
        populated_table.clear()
        assert populated_table.list_all() == {}
