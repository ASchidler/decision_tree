import time
import sys
import nonbinary.depth_cp_sat as s0
import nonbinary.depth_avellaneda_sat as s1
import nonbinary.depth_partition as s2
import nonbinary.size_narodytska as s3
import nonbinary.depth_avellaneda_base as base
from nonbinary.nonbinary_instance import ClassificationInstance, Example
from pysat.solvers import Glucose3

timeout = 3600
memlimit = 2048 * 5

enc_idx = int(sys.argv[1])

tmpdir = "./" if len(sys.argv) < 6 else sys.argv[4]
outdir = "./" if len(sys.argv) < 6 else sys.argv[5]

encodings = [
    s0,
    s1,
    s2,
    s3
]

encoding = encodings[enc_idx]

for d in range(1, 30):
    start = time.time()
    new_instance = ClassificationInstance()

    for f in range(0, d+1):
        features = []
        for f2 in range(1, d + 1):
            features.append(True if f == f2 else False)

        new_instance.add_example(Example(new_instance, features, f > 0))
    new_instance.finish()

    tree, _ = base.run(encoding, new_instance, Glucose3, slim=False, opt_size=False, timeout=timeout)

    elapsed = time.time() - start

    if tree is not None:
        print(
            f"Tree found, Nodes {tree.get_nodes()}, Depth {tree.get_depth()}, Time {elapsed}")
    else:
        print("Tree not found")
        break
