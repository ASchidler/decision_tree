import argparse as argp
import os
import resource
import time

from pysat.solvers import Glucose3
import nonbinary_instance
import incremental.entropy_strategy as es
import incremental.maintain_strategy as ms
import incremental.random_strategy as rs
import nonbinary.depth_avellaneda_sat as nbs
import nonbinary.depth_avellaneda_smt as nbt
import sat.depth_partition as dp
import sat.size_narodytska as sn
import sat.switching_encoding as se

instance_path = "nonbinary/instances"
instance_validation_path = "datasets/validate"
validation_ratios = [0, 20, 30]

# This is used for debugging, for experiments use proper memory limiting
resource.setrlimit(resource.RLIMIT_AS, (23 * 1024 * 1024 * 1024 // 2, 12 * 1024 * 1024 * 1024))

strategies = [es.EntropyStrategy2, rs.RandomStrategy, ms.MaintainingStrategy]


ap = argp.ArgumentParser(description="Python implementation for computing and improving decision trees.")
ap.add_argument("instance", type=str)
ap.add_argument("-r", dest="reduce", action="store_true", default=False,
                help="Use support set based reduction. Decreases instance size, but tree may be sub-optimal.")
ap.add_argument("-c", dest="categorical", action="store_true", default=False,
                help="Treat all features as categorical, this means a one hot encoding as in previous encodings.")
ap.add_argument("-t", dest="time_limit", action="store", default=900, type=int,
                help="The timelimit in seconds.")
ap.add_argument("-z", dest="size", action="store_true", default=False,
                help="Decrease the size as well as the depth.")
ap.add_argument("-d", dest="validation", action="store", default=0, type=int,
                help="Use data with validation set, 1=20% holdout, 2=30% holdout.")
ap.add_argument("-s", dest="use_smt", action="store_true", default=False)
ap.add_argument("-l", dest="slice", action="store", default=1, type=int, choices=[1, 2, 3, 4, 5],
                help="Which slice to use from the five cross validation sets.")

args = ap.parse_args()

if args.validation == 0:
    fls = list(x for x in os.listdir(instance_path) if x.endswith(".data"))
else:
    fls = list(
        x for x in os.listdir(instance_validation_path) if x.endswith(f"{validation_ratios[args.validation]}.data"))

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

    target_instance = fls[target_instance_idx-1][:-5]
except ValueError:
    target_instance = args.instance[:-5] if args.instance.endswith(".names") or args.instance.endswith(".data") else args.instance

print(f"{target_instance}")

start_time = time.time()
instance, test_instance, _ = nonbinary_instance.parse(instance_path, target_instance,  os.path.join(instance_path, target_instance + ".data"))

if args.categorical:
    instance.is_categorical = {x for x in range(1, instance.num_features+1)}

test_instance = instance
if os.path.exists(os.path.join(instance_path, target_instance+".test")):
    test_instance = nonbinary_instance.parse(os.path.join(instance_path, target_instance + ".test"))

tree = None

if args.use_smt:
    tree = nbt.run(instance)
else:
    tree = nbs.run(instance, Glucose3)
    # encoding.run(instance, Glucose3, timeout=args.time_limit, opt_size=args.size, check_mem=False)

if tree is None:
    print("No tree found.")
    exit(1)

#tree.check_consistency()

print(f"END Tree Depth: {tree.get_depth()}, Nodes: {tree.get_nodes()}, "
      f"Training: {tree.get_accuracy(instance.examples)}, Test: {tree.get_accuracy(test_instance.examples)}, "
      f"Time: {time.time() - start_time}")

print(tree.as_string())
