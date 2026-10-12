"""The one notebook writer behind docs/tutorials/build_*.py.

Each build script collects ("markdown" | "code", text) cells and calls
write_notebook; the JSON layout (nbformat 4.5, cell ids) lives
here only, so a fix lands in every notebook at once.
"""
import json


def write_notebook(cells, path):
    nb = {"cells": [], "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    for kind, text in cells:
        lines = text.split("\n")
        src = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "id": f"cell-{len(nb['cells']):02d}",
                "metadata": {}, "source": src}
        if kind == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        nb["cells"].append(cell)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print(f"wrote {path}: {len(cells)} cells "
          f"({sum(1 for k, _ in cells if k == 'code')} code)")
