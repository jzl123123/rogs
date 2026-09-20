#!/usr/bin/env python3
"""Compare HUGSIM function/class ASTs against pinned upstream, ignoring imports."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def definitions(text, names=None):
    return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(text).body
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and (names is None or node.name in names)}

def verify():
    entries=json.loads((ROOT/'docs/hugsim_algorithm_files.json').read_text())
    for entry in entries:
        source=ROOT/'thirdparty/HUGSIM'/entry['source']
        target=ROOT/'src/rogs/hugsim'/entry['target']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=entry['source_sha256']:
            raise ValueError('Upstream HUGSIM changed: '+entry['source'])
        if definitions(source.read_text(),entry['symbols'])!=definitions(target.read_text()):
            raise ValueError('HUGSIM algorithm changed: '+entry['target'])
    original=definitions((ROOT/'src/rogs/datasets/nusc.py').read_text(),['label2mask'])
    if original!=definitions((ROOT/'src/rogs/hugsim/shared_mask.py').read_text()):
        raise ValueError('Shared RoGS mask differs from upstream')
    return len(entries)

if __name__=='__main__':
    print(f'OK: {verify()} HUGSIM files preserve upstream functions/classes; shared mask matches RoGS')
