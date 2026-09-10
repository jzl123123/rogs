#!/usr/bin/env python3
"""Assert that packaged algorithm ASTs differ only by the rogs import namespace."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NormalizeImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith("rogs."):
            node.module = node.module[5:]
        return node


def normalized(text):
    return ast.dump(NormalizeImports().visit(ast.parse(text)), include_attributes=False)


def verify():
    files = json.loads((ROOT / "docs/algorithm_files.json").read_text())
    failures = []
    for entry in files:
        name = entry["path"]
        upstream = ROOT / "thirdparty/rogs_upstream" / name
        packaged = ROOT / "src/rogs" / name
        if hashlib.sha256(upstream.read_bytes()).hexdigest() != entry["upstream_sha256"]:
            failures.append(name + ": upstream snapshot modified")
        elif normalized(upstream.read_text()) != normalized(packaged.read_text()):
            failures.append(name + ": algorithm changed")
    if failures:
        raise ValueError("\n".join(failures))
    return len(files)


if __name__ == "__main__":
    print("OK: {} algorithm files match upstream after import normalization".format(verify()))

