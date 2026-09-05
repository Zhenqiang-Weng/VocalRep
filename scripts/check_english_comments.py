"""Reject untranslated CJK comments and docstrings in repository source files."""

import ast
import io
from pathlib import Path
import re
import subprocess
import tokenize


def main() -> None:
    """Check tracked source comments without inspecting datasets or model caches."""
    paths = subprocess.check_output(["git", "ls-files", "-z"], text=True).split("\0")
    pattern = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
    failures = []
    for name in paths:
        path = Path(name)
        if path.suffix not in {".py", ".sh", ".yaml", ".yml", ".toml"} or not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            comments = [
                (token.start[0], token.string)
                for token in tokenize.generate_tokens(io.StringIO(source).readline)
                if token.type == tokenize.COMMENT
            ]
            for node in ast.walk(ast.parse(source)):
                if isinstance(
                    node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    docstring = ast.get_docstring(node, clean=False)
                    if docstring:
                        comments.append((node.body[0].lineno, docstring))
        else:
            comments = [
                (index, line.split("#", 1)[1])
                for index, line in enumerate(source.splitlines(), 1)
                if "#" in line
            ]
        failures.extend(
            f"{path}:{line}: untranslated comment or docstring"
            for line, text in comments
            if pattern.search(text)
        )
    if failures:
        raise SystemExit("\n".join(failures))
    print("All checked comments and docstrings are free of untranslated CJK text.")


if __name__ == "__main__":
    main()
