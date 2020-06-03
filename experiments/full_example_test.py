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
import math

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

for d in range(1, 10000000, 100):
    start = time.time()
    new_instance = BddInstance()

    num_features = int(math.log2(d)) + 1

    for i in range(0, d):
        features = []
        bin_rep = bin(i)[2:][::-1]
        for c_c in bin_rep:
            features.append(c_c == "1")
        for f in range(len(features), num_features):
            features.append(False)

        ex = BddExamples(features, i % 2 == 0, i)
        new_instance.add_example(ex)

    tree, enc_size = runner.run(new_instance, encoding.new_bound(None, new_instance),
                                timeout=timeout, memlimit=memlimit)

    elapsed = time.time() - start

    if tree is not None:
        print(
            f"{d}\t{num_features}: Tree found, Nodes {tree.get_nodes()}, Depth {tree.get_depth()}, Time {elapsed}")
    else:
        print("Tree not found")
        break
