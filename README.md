# LA Shell - Linear Algebra Interpreter

LA Shell is a MATLAB-style REPL for evaluating matrix expressions. The user types a statement at a prompt, and the shell immediately tokenizes, parses, and evaluates it using a **hand-written recursive descent parser**. The design was inspired by the linear algebra demands of quantum computing.


---

## Quick Start

Clone the repository and run the REPL:

```bash
git clone https://github.com/your-username/la-shell.git
cd la-shell
pip install -r requirements.txt
python -m src.repl
```

If you don't have `git` installed, you can download the ZIP from the repository and extract it.

---

## Requirements

- Python 3.11+
- NumPy (installed automatically via `requirements.txt`)

The `requirements.txt` file includes:

```
numpy>=1.24.0
pytest>=7.0.0
pytest-cov>=4.0.0
```

To install manually:

```bash
pip install numpy
```

---

## Running the REPL

From the project root directory:

```bash
python -m src.repl
```

You will see the welcome banner:

```
LA Shell v1.2 - Linear Algebra Interpreter
Type '.help' for commands. Type '.exit' to quit.

la>
```

---

## Language Reference

### Matrix Literals

Matrices are written using square brackets. Columns within a row are separated by commas; rows are separated by semicolons. This is identical to MATLAB notation.

```
[1, 2, 3; 4, 5, 6; 7, 8, 9]
```

This produces a 3×3 matrix. A row vector is a single-row matrix:

```
[1, 2, 3]
```

A column vector is a single-column matrix:

```
[1; 2; 3]
```

Each cell element may itself be an arithmetic expression:

```
[1+1, 2*3; det(A), 0]
```

**Multi-line input.** When the opening `[` is not yet closed, the prompt changes to `...>` and the shell accumulates additional lines until all brackets are balanced:

```
la> M = [1, 2, 3;
...>     4, 5, 6;
...>     7, 8, 9]
```

### Complex Numbers

A numeric literal immediately followed by `i` is an imaginary literal, e.g. `3i`, `2.5i`, `.5i`. There is no combined real+imaginary literal syntax. Write compound complex values with ordinary addition/subtraction instead:

```
3 + 4i
2 - 5i
```

This works because `4i` is just another literal, and `+`/`-` already promote to a complex result whenever either operand is complex. This means the whole matrix is promoted to complex if any cell is complex:

```
[1+2i, 3-1i; 0+1i, 2]
```

**`i` is never a reserved word.** The imaginary suffix only applies immediately after a digit or a `.`-led number — a bare `i` with nothing in front of it always lexes as an ordinary identifier, so `i` remains free to use as a variable name (`i = 5` works exactly like any other assignment). This is a deliberate departure from MATLAB, where reassigning `i` silently breaks its meaning as `sqrt(-1)` elsewhere. The consequence is that to write the imaginary unit by itself you type `1i`, not `i`.

Output is always printed using NumPy's native `j` notation (e.g. `3+4j`), even though input uses `i` — there is currently no custom formatter to make output notation match input notation.

### Assignment

Bind the result of any expression to a named variable using `=`. Variable names must start with a letter or underscore and may contain letters, digits, and underscores.

```
A = [1, 2; 3, 4]
result = inv(A) * B
```

### Arithmetic Operators

| Operator | Meaning | Notes                                           |
|----------|---------|-------------------------------------------------|
| `+` | Matrix addition | Operands must have identical shapes             |
| `-` | Matrix subtraction | Operands must have identical shapes             |
| `*` | Matrix multiplication | Inner dimensions must agree (maps to NumPy `@`) |
| `-` (unary) | Negation | Applied to any expression                       |

Operator precedence follows standard mathematical convention: `*` binds more tightly than `+` and `-`. Parentheses can override precedence:

```
A * B + C        # (A * B) + C
A * (B + C)      # A * (B + C)
```

### Built-in Functions

| Function | Signature | Description                                                            |
|----------|-----------|------------------------------------------------------------------------|
| `transpose` | `transpose(A)` | Returns the transpose of matrix `A`                                    |
| `det` | `det(A)` | Returns the determinant of square matrix `A` as a scalar               |
| `inv` | `inv(A)` | Returns the inverse of square matrix `A` (raises an error if singular) |
| `trace` | `trace(A)` | Returns the trace (sum of diagonal elements) of `A` as a scalar        |
| `eye` | `eye(n)` | Returns the n×n identity matrix                                        |
| `zeros` | `zeros(r, c)` | Returns an r×c matrix of zeros                                         |
| `ones` | `ones(r, c)` | Returns an r×c matrix of ones                                          |
| `dagger` | `dagger(A)` | Conjugate transpose of `A` (Hermitian adjoint, `A†`)                   |
| `outer` | `outer(u, v)` | Outer product of two vectors `u` and `v`                               |
| `tensor` | `tensor(A, B)` | Tensor (Kronecker) product of `A` and `B`                              |
| `kron` | `kron(A, B)` | Alias for `tensor(A, B)`                                               |
| `commutator` | `commutator(A, B)` | `A*B - B*A` for square matrices of the same size                       |

Function arguments are full expressions, so the following are all valid:

```
inv(A * B)
eye(n + 1)
zeros(2, 4)
transpose(inv(A))
tensor(A, B)
commutator(A, B)
```
---

## Shell Commands

Commands are prefixed with a dot (`.`) and are processed before any parsing occurs.

| Command | Description |
|---------|-------------|
| `.help` | Print the list of available commands |
| `.vars` | List all currently defined variables and their values |
| `.clear` | Clear the terminal screen |
| `.save <file>` | Save the current session history to a plain-text file |
| `.load <file>` | Load and execute commands from a previously saved file |
| `.exit` | Exit the REPL (alias: `.quit`) |

---

## Example Session

```
LA Shell v1.2 — Linear Algebra Interpreter
Type '.help' for commands. Type '.exit' to quit.

la> A = [1, 2; 3, 4]
A =
[[1. 2.]
 [3. 4.]]

la> B = [5, 6; 7, 8]
B =
[[5. 6.]
 [7. 8.]]

la> C = A * B
C =
[[19. 22.]
 [43. 50.]]

la> det(A)
= -2.0

la> I = eye(2)
I =
[[1. 0.]
 [0. 1.]]

la> .vars
  A =
    [[1. 2.]
     [3. 4.]]
  B =
    [[5. 6.]
     [7. 8.]]
  C =
    [[19. 22.]
     [43. 50.]]
  I =
    [[1. 0.]
     [0. 1.]]

la> .exit
Goodbye!
```

---

## Running the Tests

```bash
python -m pytest tests/
```

To also generate an HTML coverage report:

```bash
python -m pytest tests/ --cov=src --cov-report=html
```

The target coverage threshold is 80%. Coverage reports are written to `htmlcov/index.html`.

---

## Project Structure

```
src/
├── __init__.py          Package marker
├── lexer.py             Regex-based tokenizer; produces a flat list of Token objects
├── parser.py            Recursive descent parser; builds the AST from the token list
├── evaluator.py         AST walker; executes nodes using NumPy
├── symbol_table.py      Persistent variable store mapping names to NumPy arrays
└── repl.py              REPL entry point; handles I/O, shell commands, and error display
tests/
├── __init__.py          Package marker
├── test_lexer.py        Unit tests for the Lexer
├── test_parser.py       Unit tests for the Parser and AST nodes
├── test_symbol_table.py Unit tests for the SymbolTable
└── test_evaluator.py    Unit tests for the Evaluator and built-in functions
```

---

## License

This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it for educational or research purposes.