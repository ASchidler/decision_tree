import subprocess
import sys
import os

import class_instance
import parser
import sat.size_narodytska as sn
import sat.depth_avellaneda as da
import sat.depth_partition as dp
import sat.switching_encoding as se
from pysat.solvers import Glucose3
import argparse as argp

encodings = [se, da, dp, sn]

ap = argp.ArgumentParser(description="Python implementation for computing and improving decision trees.")
ap.add_argument("instance", type=str)
ap.add_argument("-e", dest="encoding", action="store", default=0, choices=[0, 1, 2, 3], type=int,
                help="The encoding to use (0=switching, 1=Avellaneda, 2=SchidlerSzeider, 3=NarodytskaEtAl")
ap.add_argument("-m", dest="mode", action="store", default=0, choices=[0, 1, 2], type=int,
                help="The induction mode (0=All at once, 1=incremental, 2=recursive)"
                )
ap.add_argument("-r", dest="reduce", action="store_true", default=False,
                help="Use support set based reduction. Decreases instance size, but tree may be sub-optimal.")
# ap.add_argument("-a", dest="alg", action="store", default=0, choices=[0, 1, 2, 3], type=int,
#                 help="Which decision tree algorithm to use (0=C4.5, 1=ITI, 2=CART, 3=SAT).")
# ap.add_argument("-l", dest="limit_idx", action="store", default=1, type=int,
#                 help="The index for the set of limits used for selecting sub-trees.")
# ap.add_argument("-e", dest="print_tree", action="store_true", default=False,
#                 help="Export decision trees in dot format.")
# ap.add_argument("-p", dest="load_pruned", action="store", default=0,  type=int,
#                 help="Choose pruned decision tree with this index, 0 means unpruned tree.")
# ap.add_argument("-m", dest="method_prune", action="store", default=0, choices=[0, 1, 2, 3], type=int,
#                 help="Pruning method to use (0=None, 1=C4.5, 2=Cost Complexity, 3=Reduced Error).")
# ap.add_argument("-i", dest="immediate_prune", action="store_true", default=False,
#                 help="Prune each sub-tree after computation.")
# ap.add_argument("-t", dest="time_limit", action="store", default=0, type=int,
#                 help="The timelimit in seconds. Note that an elapsed timelimit will not cancel the current SAT call."
#                      "Depending on the used limits there is an imprecision of several minutes.")
# ap.add_argument("-r", dest="ratio", action="store", default=0.3, type=float,
#                 help="Ratio used for pruning. The semantics depends on the pruning method.")
# ap.add_argument("-s", dest="min_samples", action="store", default=1, type=int,
#                 help="The minimum number of samples per leaf.")
# ap.add_argument("-d", dest="validation", action="store", default=0, type=int,
#                 help="Use data with validation set, 1=20% holdout, 2=30% holdout.")

args = ap.parse_args()

instance = parser.parse(args.instance, has_header=False)
test_instance = instance
if os.path.exists(args.instance[:-4]+"test"):
    test_instance = parser.parse(args.instance[:-4] + "test")

encoding = encodings[args.encoding]

if args.reduce:
    class_instance.reduce(instance)
tree = encoding.run(instance, Glucose3)

if args.reduce:
    instance.unreduce_instance(tree)
tree.check_consistency()

print(f"Tree Depth: {tree.get_depth()}, Nodes: {tree.get_nodes()}, "
      f"Training: {tree.get_accuracy(instance.examples)}, Test: {tree.get_accuracy(test_instance.examples)}")

print(tree.as_string())

