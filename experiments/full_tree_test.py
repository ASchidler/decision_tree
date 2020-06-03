import time

import diagram_depth
import tree_depth_encoding
import diagram_encoding
import tree_node_encoding
import sys
import os
import parser
import sat_tools
import aaai_encoding
from bdd_instance import BddInstance, BddExamples

timeout = 1000
memlimit = 2048 * 5

enc_idx = int(sys.argv[1])

tmpdir = "./" if len(sys.argv) < 6 else sys.argv[4]
outdir = "./" if len(sys.argv) < 6 else sys.argv[5]

encodings = [
    diagram_encoding.DecisionDiagramEncoding,
    diagram_depth.DiagramDepthEncoding,
    tree_depth_encoding.TreeDepthEncoding,
    tree_node_encoding.TreeEncoding,
    aaai_encoding.AAAIEncoding
]

encoding = encodings[enc_idx]
solver = sat_tools.GlucoseSolver
runner = sat_tools.SatRunner(encoding, solver(), base_path=tmpdir)

for d in range(1, 30):
    start = time.time()
    new_instance = BddInstance()

    for f in range(0, d+1):
        features = []
        for f2 in range(1, d + 1):
            features.append(True if f == f2 else False)

        ex = BddExamples(features, f > 0, d)
        new_instance.add_example(ex)

    tree, enc_size = runner.run(new_instance, encoding.new_bound(None, new_instance),
                                timeout=timeout, memlimit=memlimit)

    elapsed = time.time() - start

    if tree is not None:
        print(
            f"Tree found, Nodes {tree.get_nodes()}, Depth {tree.get_depth()}, Time {elapsed}")
    else:
        print("Tree not found")
        break
