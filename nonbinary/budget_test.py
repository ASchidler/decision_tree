import argparse as argp
import os
import random
import resource
import sys
import time

from pysat.solvers import Glucose3

import incremental.entropy_strategy as es
import incremental.maintain_strategy as ms
import incremental.random_strategy as rs
import nonbinary.depth_avellaneda_base as base
import nonbinary.depth_avellaneda_sat as nbs
import nonbinary.depth_avellaneda_sat2 as nbs2
import nonbinary.depth_avellaneda_sat3 as nbs3
import nonbinary.depth_avellaneda_smt as nbt
import nonbinary_instance

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
ap.add_argument("-u", dest="maintain", action="store_true", default=False,
                help="Force maintaining of sizes for SLIM.")

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

instance, test_instance, validation_instance = nonbinary_instance.parse(instance_path, target_instance, args.slice, use_validation=False)

if args.use_smt:
    enc = nbt
else:
    if args.hybrid:
        enc = nbs3
    elif args.alt_sat:
        enc = nbs2
    else:
        enc = nbs

if args.reduce:
    instance.reduce_with_key()

for i in range(50, len(instance.examples), 50):
    to = 0
    for _ in range(0, 3):
        new_instance = nonbinary_instance.ClassificationInstance()

        for _ in range(0, i):
            new_instance.add_example(instance.examples[random.randint(0, len(instance.examples)-1)].copy(new_instance))
        new_instance.finish()
        start = time.time()
        if not args.use_smt:
            tree = base.run(enc, new_instance, Glucose3, slim=False, opt_size=args.size, timeout=600)
        else:
            tree = nbt.run(new_instance, opt_size=args.size, timeout=600)

        if tree:
            esize = enc.estimate_size(new_instance, tree.get_depth())
            print(f"E:{i} T:{time.time()-start} C:{len(instance.classes)} F:{new_instance.num_features} DS:{sum(len(new_instance.domains[x]) for x in range(1, new_instance.num_features+1))}"
                  f" DM:{max(len(new_instance.domains[x]) for x in range(1, new_instance.num_features+1))} D:{tree.get_depth()} S:{esize}")
        else:
            print(f"E: {i} Time: -1")
            to += 1
    if to == 3:
        break


