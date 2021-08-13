import argparse as argp
import os
import resource
import sys
import time
import improve_strategy
import tree_parsers

from pysat.solvers import Glucose3
import nonbinary_instance
import incremental.entropy_strategy as es
import incremental.maintain_strategy as ms
import incremental.random_strategy as rs
import nonbinary.depth_avellaneda_sat as nbs
import nonbinary.depth_avellaneda_sat2 as nbs2
import nonbinary.depth_avellaneda_sat3 as nbs3
import nonbinary.depth_avellaneda_smt as nbt
import nonbinary.depth_avellaneda_base as base
from threading import Timer

instance_path = "nonbinary/instances"
instance_validation_path = "datasets/validate"

# This is used for debugging, for experiments use proper memory limiting
resource.setrlimit(resource.RLIMIT_AS, (23 * 1024 * 1024 * 1024 // 2, 12 * 1024 * 1024 * 1024))

strategies = [es.EntropyStrategy2, rs.RandomStrategy, ms.MaintainingStrategy]


ap = argp.ArgumentParser(description="Python implementation for computing and improving decision trees.")
ap.add_argument("instance", type=str)
ap.add_argument("-r", dest="reduce", action="store_true", default=False,
                help="Use support set based reduction. Decreases instance size, but tree may be sub-optimal.")
ap.add_argument("-c", dest="categorical", action="store_true", default=False,
                help="Treat all features as categorical, this means a one hot encoding as in previous encodings.")
ap.add_argument("-t", dest="time_limit", action="store", default=0, type=int,
                help="The timelimit in seconds.")
ap.add_argument("-z", dest="size", action="store_true", default=False,
                help="Decrease the size as well as the depth.")
ap.add_argument("-e", dest="slim_opt", action="store_true", default=False,
                help="Optimize away extension leaves.")
ap.add_argument("-d", dest="validation", action="store_true", default=False,
                help="Use data with validation set.")
ap.add_argument("-s", dest="use_smt", action="store_true", default=False)
ap.add_argument("-l", dest="slice", action="store", default=1, type=int,
                help="Which slice to use from the five cross validation sets.")
ap.add_argument("-a", dest="alt_sat", action="store_true", default=False,
                help="Use alternative SAT encoding.")
ap.add_argument("-y", dest="hybrid", action="store_true", default=False,
                help="Use hybrid mode, allowing for <= and =.")
ap.add_argument("-w", dest="weka", action="store_false", default=True,
                help="Use CART instead of WEKA trees.")
ap.add_argument("-m", dest="slim", action="store_true", default=False,
                help="Use local improvement instead of exact results.")
ap.add_argument("-u", dest="multiclass", action="store_true", default=False,
                help="For mid-reductions allow multiclass.")

args = ap.parse_args()

fls = list(x for x in os.listdir(instance_path) if x.endswith(".data"))

fls.sort()

# if args.choices:
#     for i, cf in enumerate(fls):
#         print(f"{i+1}: {cf}")
#     exit(0)

try:
    target_instance_idx = int(args.instance)

    if target_instance_idx > len(fls):
        print(f"Only {len(fls)} files are known.")
        exit(1)

    parts = fls[target_instance_idx-1][:-5].split(".")
    target_instance = ".".join(parts[:-1])
    args.slice = int(parts[-1])

except ValueError:
    target_instance = args.instance[:-5] if args.instance.endswith(".names") or args.instance.endswith(".data") else args.instance

print(f"Instance: {target_instance}, {args}")
sys.stdout.flush()

start_time = time.time()

def exit_timeout():
    # TODO: There is a race condition in case the tree is currently changing. This should rarely be the case, as
    # this takes a lot shorter than reduction +
    print(f"Timeout: {time.time() - start_time}")
    #tree.clean(instance, min_samples=args.min_samples)
    print(f"Time: End\t\t"
          f"Training {tree.get_accuracy(instance.examples):.4f}\t"
          f"Test {tree.get_accuracy(test_instance.examples):.4f}\t"
          f"Depth {tree.get_depth():03}\t"
          f"Nodes {tree.get_nodes()}")
    print(tree.as_string())
    sys.stdout.flush()
    exit(1)

instance, test_instance, validation_instance = nonbinary_instance.parse(instance_path, target_instance, args.slice)

timer = None
if args.time_limit > 0:
    timer = Timer(args.time_limit * 1.1 - (time.time() - start_time), exit_timeout)
    timer.start()

if args.reduce:
    print(f"{instance.num_features}, {len(instance.examples)}")
    instance.reduce_with_key()
    print(f"{instance.num_features}, {len(instance.examples)}")

if args.categorical:
    instance.is_categorical = {x for x in range(1, instance.num_features+1)}

tree = None
if args.use_smt:
    enc = nbt
else:
    if args.hybrid:
        enc = nbs3
    elif args.alt_sat:
        enc = nbs2
    else:
        enc = nbs

if args.slim:
    algo = "w" if args.weka else "c"
    tree = tree_parsers.parse_internal_tree(f"nonbinary/results/trees/unpruned/{target_instance}.{args.slice}.{algo}.dt")
    print(f"START Tree Depth: {tree.get_depth()}, Nodes: {tree.get_nodes()}, "
          f"Training: {tree.get_accuracy(instance.examples)}, Test: {tree.get_accuracy(test_instance.examples)}, "
          f"Time: {time.time() - start_time}")

    improve_strategy.run(tree, instance, test_instance, Glucose3, enc, timelimit=args.time_limit, opt_size=args.size, opt_slim=args.slim_opt, multiclass=args.multiclass)
else:
    if not args.use_smt:
        tree = base.run(enc, instance, Glucose3, slim=False, opt_size=args.size)
    else:
        tree = nbt.run(instance)

if tree is None:
    print("No tree found.")
    exit(1)

#tree.check_consistency()

instance.unreduce(tree)
print(f"{instance.num_features}, {len(instance.examples)}")
print(f"END Tree Depth: {tree.get_depth()}, Nodes: {tree.get_nodes()}, "
      f"Training: {tree.get_accuracy(instance.examples)}, Test: {tree.get_accuracy(test_instance.examples)}, "
      f"Time: {time.time() - start_time}")

print(tree.as_string())

if timer is not None:
    timer.cancel()
