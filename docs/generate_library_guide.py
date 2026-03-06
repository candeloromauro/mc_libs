#!/usr/bin/env python3
"""Generate a LaTeX guide for all mc_* modules and public APIs."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class FunctionDoc:
    name: str
    signature: str
    description: str
    args: str
    returns: str
    raises: str
    examples: str


@dataclass
class ClassDoc:
    name: str
    description: str
    methods: list[FunctionDoc]


@dataclass
class ModuleDoc:
    module_name: str
    path: Path
    description: str
    functions: list[FunctionDoc]
    classes: list[ClassDoc]


SECTION_HEADERS = {"Args:": "args", "Returns:": "returns", "Raises:": "raises", "Examples:": "examples", "Example:": "examples"}


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def compact(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def sanitize_text(text: str) -> str:
    replacements = {
        "→": "->",
        "√": "sqrt",
        "≤": "<=",
        "≥": ">=",
        "×": "x",
        "–": "-",
        "—": "-",
        "π": "pi",
        "°": " deg",
        "γ": "gamma",
        "∇": "nabla",
        "τ": "tau",
        "σ": "sigma",
        "θ": "theta",
        "φ": "phi",
        "ψ": "psi",
        "Δ": "Delta",
        "μ": "mu",
        "α": "alpha",
        "β": "beta",
        "λ": "lambda",
        "η": "eta",
        "ρ": "rho",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("ascii", errors="replace").decode("ascii")


def parse_doc_sections(doc: str | None) -> dict[str, str]:
    if not doc:
        return {"description": "No description provided.", "args": "", "returns": "", "raises": "", "examples": ""}

    cleaned = [line.rstrip() for line in inspect.cleandoc(doc).splitlines()]

    sections = {"description": [], "args": [], "returns": [], "raises": [], "examples": []}
    current = "description"

    for raw in cleaned:
        stripped = raw.strip()
        if stripped in SECTION_HEADERS:
            current = SECTION_HEADERS[stripped]
            continue
        sections[current].append(raw)

    out = {k: compact("\n".join(v)) for k, v in sections.items()}
    if not out["description"]:
        out["description"] = "No description provided."
    return out


def collect_def_signature(source_lines: list[str], node: ast.FunctionDef) -> str:
    start = node.lineno - 1
    out: list[str] = []
    for i in range(start, len(source_lines)):
        line = source_lines[i].rstrip("\n")
        out.append(line)
        if line.rstrip().endswith(":"):
            break
    joined = "\n".join(out).strip()
    if joined.startswith("def "):
        joined = joined[4:]
    if joined.endswith(":"):
        joined = joined[:-1]
    return joined.strip()


def function_doc_from_node(source_lines: list[str], node: ast.FunctionDef) -> FunctionDoc:
    doc = ast.get_docstring(node) or ""
    sections = parse_doc_sections(doc)
    return FunctionDoc(
        name=node.name,
        signature=collect_def_signature(source_lines, node),
        description=sections["description"],
        args=sections["args"],
        returns=sections["returns"],
        raises=sections["raises"],
        examples=sections["examples"],
    )


def module_doc(path: Path, root: Path) -> ModuleDoc:
    source = path.read_text(encoding="utf-8")
    source_lines = source.splitlines(keepends=True)
    tree = ast.parse(source)

    rel = path.relative_to(root)
    if rel.name == "__init__.py":
        module_name = ".".join(rel.parts[:-1])
    else:
        module_name = ".".join(rel.with_suffix("").parts)
    module_description = parse_doc_sections(ast.get_docstring(tree) or "")["description"]

    functions: list[FunctionDoc] = []
    classes: list[ClassDoc] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            functions.append(function_doc_from_node(source_lines, node))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            class_doc = parse_doc_sections(ast.get_docstring(node) or "")
            methods: list[FunctionDoc] = []
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and not child.name.startswith("_"):
                    methods.append(function_doc_from_node(source_lines, child))
            classes.append(ClassDoc(name=node.name, description=class_doc["description"], methods=methods))

    return ModuleDoc(
        module_name=module_name,
        path=path,
        description=module_description,
        functions=functions,
        classes=classes,
    )


def module_files(root: Path) -> Iterable[Path]:
    for pkg in sorted(root.iterdir()):
        if not (pkg.is_dir() and pkg.name.startswith("mc_") and (pkg / "__init__.py").exists()):
            continue
        for py_file in sorted(pkg.rglob("*.py")):
            if "__pycache__" in py_file.parts or "docs" in py_file.parts:
                continue
            yield py_file


def add_block(lines: list[str], title: str, content: str) -> None:
    lines.append(f"\\paragraph{{{latex_escape(title)}}}")
    if content:
        content = sanitize_text(inspect.cleandoc(content))
        lines.append("\\begin{lstlisting}[language=Python]")
        lines.extend(content.splitlines())
        lines.append("\\end{lstlisting}")
    else:
        lines.append("Not specified.")


def build_tree_lines(docs: list[ModuleDoc]) -> list[tuple[int, str]]:
    by_pkg: dict[str, list[ModuleDoc]] = {}
    for doc in docs:
        pkg = doc.module_name.split(".")[0]
        by_pkg.setdefault(pkg, []).append(doc)

    tree: list[tuple[int, str]] = [(0, "mc_libs")]
    pkgs = sorted(by_pkg.keys())
    for pkg_idx, pkg in enumerate(pkgs):
        pkg_last = pkg_idx == len(pkgs) - 1
        pkg_branch = "`-" if pkg_last else "|-"
        tree.append((1, f"{pkg_branch} {pkg}"))
        pkg_prefix = "   " if pkg_last else "|  "

        mods = sorted(by_pkg[pkg], key=lambda x: x.module_name)
        for mod_idx, mod in enumerate(mods):
            mod_last = mod_idx == len(mods) - 1
            mod_branch = "`-" if mod_last else "|-"
            tree.append((2, f"{pkg_prefix}{mod_branch} {mod.module_name}"))
            mod_prefix = f"{pkg_prefix}{'   ' if mod_last else '|  '}"

            callable_names: list[str] = []
            callable_names.extend(fn.name for fn in mod.functions)
            for cls in mod.classes:
                if not cls.methods:
                    callable_names.append(cls.name)
                else:
                    callable_names.extend(f"{cls.name}.{m.name}" for m in cls.methods)

            callable_names = sorted(callable_names)
            for call_idx, name in enumerate(callable_names):
                call_last = call_idx == len(callable_names) - 1
                call_branch = "`-" if call_last else "|-"
                tree.append((3, f"{mod_prefix}{call_branch} {name}"))

    return tree


def build_tex(docs: list[ModuleDoc], out_path: Path) -> None:
    packages = sorted({doc.module_name.split(".")[0] for doc in docs})

    lines: list[str] = []
    lines.extend(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{hyperref}",
            r"\usepackage{listings}",
            r"\usepackage{xcolor}",
            r"\definecolor{TreeLevelZero}{HTML}{1E1E1E}",
            r"\definecolor{TreeLevelOne}{HTML}{1F5C99}",
            r"\definecolor{TreeLevelTwo}{HTML}{0F7B6C}",
            r"\definecolor{TreeLevelThree}{HTML}{8A3FA0}",
            r"\lstset{",
            r"  basicstyle=\ttfamily\small,",
            r"  frame=single,",
            r"  breaklines=true,",
            r"  columns=fullflexible",
            r"}",
            r"\title{mc\_libs Comprehensive Library Guide}",
            rf"\date{{Generated on {latex_escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}}}",
            r"\begin{document}",
            r"\maketitle",
            r"\tableofcontents",
            r"\newpage",
            r"This document is generated from source modules and docstrings under the mc\_libs repository.",
            r"",
            r"\section{Packages Overview}",
            r"\begin{itemize}",
        ]
    )

    for pkg in packages:
        lines.append(rf"\item \texttt{{{latex_escape(pkg)}}}")
    lines.append(r"\end{itemize}")
    lines.append("")
    lines.append(r"\section{Library Tree}")
    lines.append(r"\begingroup")
    lines.append(r"\ttfamily")
    tree_colors = {
        0: "TreeLevelZero",
        1: "TreeLevelOne",
        2: "TreeLevelTwo",
        3: "TreeLevelThree",
    }
    for level, raw_line in build_tree_lines(docs):
        color_name = tree_colors.get(level, "TreeLevelZero")
        lines.append(rf"\textcolor{{{color_name}}}{{{latex_escape(raw_line)}}}\\")
    lines.append(r"\endgroup")

    by_pkg: dict[str, list[ModuleDoc]] = {}
    for doc in docs:
        pkg = doc.module_name.split(".")[0]
        by_pkg.setdefault(pkg, []).append(doc)

    for pkg in sorted(by_pkg):
        lines.append("")
        lines.append(rf"\section{{Package \texttt{{{latex_escape(pkg)}}}}}")

        for mod in sorted(by_pkg[pkg], key=lambda x: x.module_name):
            lines.append(rf"\subsection{{Module \texttt{{{latex_escape(mod.module_name)}}}}}")
            lines.append(latex_escape(sanitize_text(mod.description)))
            lines.append("")

            if not mod.functions and not mod.classes:
                lines.append("No public functions or classes were found in this module.")
                lines.append("")
                continue

            for fn in mod.functions:
                lines.append(rf"\subsubsection{{Function \texttt{{{latex_escape(fn.name)}}}}}")
                lines.append(latex_escape(sanitize_text(fn.description)))
                lines.append("")
                add_block(lines, "Syntax", f"from {mod.module_name} import {fn.name}\n{fn.signature}")
                add_block(lines, "Inputs", fn.args)
                add_block(lines, "Outputs", fn.returns)
                add_block(lines, "Raises", fn.raises)
                add_block(lines, "Example", fn.examples)
                lines.append("")

            for cls in mod.classes:
                lines.append(rf"\subsubsection{{Class \texttt{{{latex_escape(cls.name)}}}}}")
                lines.append(latex_escape(sanitize_text(cls.description)))
                lines.append("")

                if not cls.methods:
                    lines.append("No public methods were found.")
                    lines.append("")
                    continue

                for method in cls.methods:
                    lines.append(rf"\paragraph{{Method \texttt{{{latex_escape(method.name)}}}}}")
                    lines.append(latex_escape(sanitize_text(method.description)))
                    lines.append("")
                    add_block(lines, "Syntax", method.signature)
                    add_block(lines, "Inputs", method.args)
                    add_block(lines, "Outputs", method.returns)
                    add_block(lines, "Raises", method.raises)
                    add_block(lines, "Example", method.examples)
                    lines.append("")

    lines.append(r"\end{document}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    docs_dir = Path(__file__).resolve().parent
    repo_root = docs_dir.parent

    docs = [module_doc(path, repo_root) for path in module_files(repo_root)]
    out_path = docs_dir / "mc_libs_guide.tex"
    build_tex(docs, out_path)
    print(f"[write] {out_path}")
    print(f"[info] modules documented: {len(docs)}")


if __name__ == "__main__":
    main()
