#!/usr/bin/env python3
import json
import sys
from pathlib import Path

p = Path('/home/workhorse/utils/ai_scientist/AI-Scientist-v2/experiments/2025-09-27_20-17-54_cylindrical_plastic_solar_cells_attempt_0/logs/0-run/stage_1_initial_implementation_1_preliminary/journal.json')
if not p.exists():
    print('journal.json not found at', p)
    sys.exit(2)

with p.open() as f:
    data = json.load(f)

nodes = data.get('nodes') or []
print(f'Total nodes in journal: {len(nodes)}')

good_nodes = []
for n in nodes:
    nid = n.get('id')
    is_buggy = n.get('is_buggy')
    is_buggy_plots = n.get('is_buggy_plots')
    metric = n.get('metric')
    parent = n.get('parent_id')
    children = n.get('children') or []
    print(f"- id={nid} parent={parent} children={len(children)} is_buggy={is_buggy} is_buggy_plots={is_buggy_plots} metric={metric}")
    if is_buggy is False and (is_buggy_plots is False or is_buggy_plots is None):
        good_nodes.append(n)

print('\nGood nodes count:', len(good_nodes))
if len(good_nodes) > 0:
    print('Sample good node ids:', [n.get('id') for n in good_nodes[:5]])
else:
    print('No good nodes found')
