import time

import nonbinary.depth_avellaneda_base as base
import nonbinary.depth_avellaneda_sat as nbs
import nonbinary.depth_avellaneda_sat2 as nbs2
import nonbinary.depth_avellaneda_sat3 as nbs3
import nonbinary.depth_partition as nbp
import nonbinary.nonbinary_instance
import nonbinary.size_narodytska as nbn
import nonbinary.depth_avellaneda_smt2 as nbt
import nonbinary.depth_switching as nbw
import nonbinary.improve_strategy as improve_strategy
import nonbinary.depth_cp_sat as cps
import nonbinary.nonbinary_instance as nb_instance
import tree_parsers
import sys
import os
import nonbinary.depth_avellaneda_base as ab
from pysat.solvers import Glucose3
import random

enc = [nbs, nbs2, nbs3, nbt, nbp, nbn, nbw, cps][int(sys.argv[1])]
abs_path = os.path.split(os.path.realpath(__file__))[0]

instance_path = os.path.join(abs_path, "instances")

instances = [x for x in os.listdir(instance_path) if x.endswith(".data") and x.endswith(".1.data")]
instances.sort()

instance_name = instances[int(sys.argv[2])]
print(f"Encoding: {sys.argv[1]}")
print(f"Instance: {instance_name}")

instance, _, _ = nb_instance.parse(instance_path, instance_name[:-7], 1, False, False)

print(f"Samples: {len(instance.examples)}")
print(f"Features: {instance.num_features}")

tree = tree_parsers.parse_internal_tree(os.path.join(abs_path, f"results/trees/unpruned/{instance_name[:-5]}.c.dt"))
print(f"Nodes: {tree.get_nodes()}")
assigned = tree.assign(instance)

if sys.argv[3] == "1":
    for c_node in tree.nodes:
        if c_node and c_node.is_leaf:
            c_parent = c_node.parent.parent
            solved = True
            depth = 2
            while c_parent and solved:
                solved = False
                sub_instance = nb_instance.ClassificationInstance()
                for c_e in assigned[c_parent.id]:
                    sub_instance.add_example(c_e)

                before = time.time()
                result = ab.run(enc, sub_instance, Glucose3, start_bound=1, timeout=1200, ub=depth, log=True, slim=False)

                if time.time() - before < 1200:
                    solved = True
                    c_parent = c_parent.parent
                    depth += 1
else:
    max_tries = 10
    for c_sample_size in [10, 20, 40, 50, 75, 100, 150, 200, 250, 300, 250, 400, 450, 500, 600, 700, 800, 900, 1000, 1100, 1200,
                          1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]:
        if c_sample_size > len(instance.examples):
            break

        success_cnt = 0
        for _ in range(0, max_tries):
            selection = random.sample(instance.examples, c_sample_size)
            sub_instance = nb_instance.ClassificationInstance()
            for ce in selection:
                sub_instance.add_example(ce)

            result = ab.run(enc, sub_instance, Glucose3, start_bound=1, timeout=1200, log=True, slim=False)
            if result:
                success_cnt += 1

        if success_cnt == 0:
            break
