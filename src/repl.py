"""REPL (Read-Eval-Print Loop) for the Linear Algebra Shell.

This module is the sole entry point for interactive use.  It implements:

* ANSI color helpers (green for results, red for errors, white for prompts).
* Multi-line input detection: unmatched ``[`` brackets cause the prompt to
  switch from ``la> `` to ``...> `` until brackets are balanced.
* Shell commands prefixed with ``.``: ``.help``, ``.vars``, ``.clear``,
  ``.save <file>``, ``.load <file>``, ``.exit`` / ``.quit``.
* A never-crashing evaluation wrapper that catches ``LexerError``,
  ``SyntaxError``, ``NameError``, ``TypeError``, ``ValueError``, and
  ``RuntimeError``.

Run directly::

    python -m src.repl

or::

    python src/repl.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

from src.evaluator import Evaluator
from src.lexer import Lexer, LexerError
from src.parser import AssignmentNode, Parser
from src.symbol_table import SymbolTable

# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

_GREEN = "\033[92m"
_RED   = "\033[91m"
_WHITE = "\033[97m"
_RESET = "\033[0m"


def green(text: str) -> str:
    """Wrap *text* in the ANSI green escape sequence.

    Args:
        text: Plain string to colorize.

    Returns:
        String surrounded by the green ANSI start and reset codes.
    """
    return f"{_GREEN}{text}{_RESET}"


def red(text: str) -> str:
    """Wrap *text* in the ANSI red escape sequence.

    Args:
        text: Plain string to colorize.

    Returns:
        String surrounded by the red ANSI start and reset codes.
    """
    return f"{_RED}{text}{_RESET}"


def white(text: str) -> str:
    """Wrap *text* in the ANSI bright-white escape sequence.

    Args:
        text: Plain string to colorize.

    Returns:
        String surrounded by the bright-white ANSI start and reset codes.
    """
    return f"{_WHITE}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Bracket depth helper
# ---------------------------------------------------------------------------

def _unmatched_open_brackets(text: str) -> int:
    """Return the number of unmatched ``[`` characters in *text*.

    Scans *text* character by character, incrementing a depth counter on
    ``[`` and decrementing on ``]`` (clamped to zero so that an extra ``]``
    is not counted as negative depth).

    Args:
        text: Accumulated input text, potentially spanning multiple lines.

    Returns:
        Non-negative integer; zero means all brackets are balanced.
    """
    depth = 0
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
    return depth


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _scalar_str(value: float) -> str:
    """Format a scalar float without redundant trailing zeros.

    Uses Python's ``%g`` conversion which suppresses trailing zeros and
    switches to scientific notation only when necessary.

    Args:
        value: The scalar value to format.

    Returns:
        A compact string representation (e.g. ``"3"`` not ``"3.0"``).
    """
    try:
        return f"{value:g}"
    except (ValueError, TypeError):
        return str(value)


def _format_result(ast_root: object, result: np.ndarray) -> str:
    """Build the display string for an evaluation result.

    For assignment nodes the variable name is included as a prefix.
    A 1×1 array is displayed as a scalar; larger arrays use NumPy's
    default ``__str__`` formatting.

    Args:
        ast_root: The root AST node that produced *result*.  Used only to
            detect ``AssignmentNode`` and extract the variable name.
        result: The NumPy array returned by the evaluator.

    Returns:
        A formatted string ready to be passed to ``print()``.
    """
    prefix = f"{ast_root.name} " if isinstance(ast_root, AssignmentNode) else ""

    if result.size == 1:
        return f"{prefix}= {_scalar_str(result.item())}"

    return f"{prefix}=\n{result}"


# ---------------------------------------------------------------------------
# Core evaluate-and-print helper
# ---------------------------------------------------------------------------

def _evaluate_and_print(source: str, evaluator: Evaluator) -> None:
    """Tokenize, parse, evaluate *source*, and print a colorized result.

    All known exception types are caught and printed as red error messages
    so that the REPL loop is never interrupted by evaluation failures.

    Args:
        source: A complete (possibly multi-line) LA Shell statement.
        evaluator: The active ``Evaluator`` for this session.
    """
    try:
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        result = evaluator.eval(ast)
        print(green(_format_result(ast, result)))
    except LexerError as exc:
        print(red(f"LexerError: {exc}"))
    except SyntaxError as exc:
        print(red(f"SyntaxError: {exc}"))
    except NameError as exc:
        print(red(f"NameError: {exc}"))
    except TypeError as exc:
        print(red(f"TypeError: {exc}"))
    except ValueError as exc:
        print(red(f"ValueError: {exc}"))
    except RuntimeError as exc:
        print(red(f"RuntimeError: {exc}"))
    except Exception as exc:  # pragma: no cover — genuine unexpected errors
        print(red(f"Error: {exc}"))


# ---------------------------------------------------------------------------
# File-replay helper (used by .load)
# ---------------------------------------------------------------------------

def _replay_file(
    lines: list[str],
    evaluator: Evaluator,
    history: list[str],
) -> None:
    """Execute lines loaded from a file with bracket-aware accumulation.

    Processes *lines* in order, using the same multi-line bracket detection
    as the interactive loop so that matrix literals that span several lines
    in the saved file are handled correctly.  Each logical statement is
    echoed to stdout before evaluation.

    Args:
        lines: Individual text lines read from the file (no newline suffix).
        evaluator: The active evaluator for this session.
        history: Session history list; executed statements are appended.
    """
    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        idx += 1

        stripped = raw.strip()
        # Skip blank lines and comment lines (lines starting with '#').
        if not stripped or stripped.startswith("#"):
            continue

        # Accumulate until brackets are balanced.
        accumulated = raw
        while _unmatched_open_brackets(accumulated) > 0 and idx < len(lines):
            accumulated += "\n" + lines[idx]
            idx += 1

        if not accumulated.strip():
            continue

        # Echo the statement with appropriate prompts.
        echo_lines = accumulated.splitlines()
        print(white("la> ") + echo_lines[0])
        for cont_line in echo_lines[1:]:
            print(white("...> ") + cont_line)

        history.append(accumulated)
        _evaluate_and_print(accumulated, evaluator)


# ---------------------------------------------------------------------------
# Shell command handler
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
Available commands:
  .help           Show this help message
  .vars           List all defined variables and their current values
  .clear          Clear the terminal screen
  .save <file>    Save session history to a plain-text file
  .load <file>    Load and execute commands from a previously saved file
  .exit           Exit the REPL  (alias: .quit)"""


def _handle_command(
    line: str,
    symbol_table: SymbolTable,
    evaluator: Evaluator,
    history: list[str],
) -> bool:
    """Dispatch a dot-prefixed shell command.

    Args:
        line: The trimmed command line, e.g. ``'.help'`` or ``'.save out.la'``.
        symbol_table: The session variable store (read by ``.vars``).
        evaluator: The active evaluator (used when replaying ``.load`` files).
        history: Session history list (read/written by ``.save`` / ``.load``).

    Returns:
        ``True`` if the REPL should exit after this command, ``False``
        otherwise.
    """
    parts = line.split()
    cmd = parts[0][1:]  # Strip leading '.'

    if cmd == "help":
        print(_HELP_TEXT)

    elif cmd == "vars":
        bindings = symbol_table.list_all()
        if not bindings:
            print("  (no variables defined)")
        else:
            for var_name, value in bindings.items():
                if value.size == 1:
                    print(f"  {var_name} = {_scalar_str(value.item())}")
                else:
                    print(f"  {var_name} =")
                    for mat_line in str(value).splitlines():
                        print(f"    {mat_line}")

    elif cmd == "clear":
        os.system("cls" if os.name == "nt" else "clear")

    elif cmd == "save":
        if len(parts) < 2:
            print(red("Error: .save requires a filename argument, e.g. '.save session.la'"))
        else:
            filename = parts[1]
            try:
                with open(filename, "w", encoding="utf-8") as fh:
                    for entry in history:
                        fh.write(entry)
                        if not entry.endswith("\n"):
                            fh.write("\n")
                print(f"Session saved to {filename!r} ({len(history)} statement(s)).")
            except OSError as exc:
                print(red(f"Error: Could not write to {filename!r}: {exc}"))

    elif cmd == "load":
        if len(parts) < 2:
            print(red("Error: .load requires a filename argument, e.g. '.load session.la'"))
        else:
            filename = parts[1]
            try:
                with open(filename, "r", encoding="utf-8") as fh:
                    file_lines = fh.read().splitlines()
                _replay_file(file_lines, evaluator, history)
            except OSError as exc:
                print(red(f"Error: Could not read {filename!r}: {exc}"))

    elif cmd in ("exit", "quit"):
        return True

    else:
        print(red(f"Error: Unknown command '.{cmd}'. Type '.help' for a list of commands."))

    return False


# ---------------------------------------------------------------------------
# Main REPL loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the LA Shell interactive REPL.

    Creates a fresh ``SymbolTable`` and ``Evaluator``, prints a welcome
    banner, then enters the main read-eval-print loop.

    The loop:

    1. Reads a line with the ``la> `` primary prompt.
    2. If the line starts with ``.``, dispatches it as a shell command.
    3. Otherwise, accumulates continuation lines (``...> `` prompt) until
       all ``[`` brackets are matched.
    4. Passes the complete statement to ``_evaluate_and_print``.
    5. Records non-command statements in ``history`` for ``.save``.

    The loop **never raises an unhandled exception**.  ``EOFError`` (piped
    input exhausted) and ``KeyboardInterrupt`` (Ctrl-C) both trigger a
    clean exit with a farewell message.
    """
    symbol_table = SymbolTable()
    evaluator = Evaluator(symbol_table)
    history: list[str] = []

    print(white("LA Shell v1.2 - Linear Algebra Interpreter"))
    print(white("Type '.help' for commands. Type '.exit' to quit.\n"))

    while True:
        # ------------------------------------------------------------------
        # Read primary line
        # ------------------------------------------------------------------
        try:
            raw = input(white("la> "))
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        line = raw.strip()
        if not line:
            continue

        # ------------------------------------------------------------------
        # Shell commands
        # ------------------------------------------------------------------
        if line.startswith("."):
            try:
                should_exit = _handle_command(
                    line, symbol_table, evaluator, history
                )
            except Exception as exc:  # pragma: no cover
                print(red(f"Error: {exc}"))
                should_exit = False
            if should_exit:
                break
            continue

        # ------------------------------------------------------------------
        # Multi-line accumulation
        # ------------------------------------------------------------------
        accumulated = raw  # Preserve original whitespace for the parser.
        try:
            while _unmatched_open_brackets(accumulated) > 0:
                try:
                    next_raw = input(white("...> "))
                except EOFError:
                    print()
                    break
                except KeyboardInterrupt:
                    print()
                    accumulated = ""
                    break
                accumulated += "\n" + next_raw
        except Exception as exc:  # pragma: no cover
            print(red(f"Input error: {exc}"))
            continue

        if not accumulated.strip():
            continue

        # Record in history before evaluation so it is saved even on error.
        history.append(accumulated)

        # ------------------------------------------------------------------
        # Evaluate
        # ------------------------------------------------------------------
        _evaluate_and_print(accumulated, evaluator)

    print(white("\nGoodbye!"))


if __name__ == "__main__":
    main()
