"""Keep executable setup and deliberately unfinished sections readable."""

from pathlib import Path

import nbformat


def test_notebook_structure_and_python_syntax() -> None:
    for path in Path("notebooks").glob("*.ipynb"):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, str(path), "exec")
    evaluation = nbformat.read("notebooks/04_evaluation.ipynb", as_version=4)
    placeholders = [cell.source for cell in evaluation.cells if cell.cell_type == "code"]
    assert sum("# TODO:" in source for source in placeholders) == 4
