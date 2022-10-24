import sys
import os
import time
from collections import defaultdict

from nonbinary import nonbinary_instance
import random

instance_path = "nonbinary/instances"

target_instance_idx = int(sys.argv[1])
target_heuristic = sys.argv[2]

fls = list(x for x in os.listdir(instance_path) if x.endswith(".data"))
fls.sort()
parts = fls[target_instance_idx-1][:-5].split(".")
target_instance = ".".join(parts[:-1])

instance, _, _ = nonbinary_instance.parse(instance_path, target_instance, 1, use_validation=False, use_test=False)

print(f"{target_instance} {target_heuristic} {instance.num_features}")

for cl in range(min(1000, len(instance.examples)-1), len(instance.examples), 50):
    local_instance = nonbinary_instance.ClassificationInstance()
    examples = random.sample(instance.examples, cl)

    classes = defaultdict(int)
    for c_e in examples:
        local_instance.add_example(c_e)
        classes[c_e.cls] += 1
    local_instance.finish()

    start_time = time.time()

    if target_heuristic == "1":
        local_instance.min_key_greedy2(False, False)
    elif target_heuristic == "0":
        local_instance.min_key_greedy(False, False)
    else:
        local_instance.min_key_random(False, False)

    expected_time = local_instance.num_features
    for cv in classes.values():
        expected_time *= cv

    print(f"{cl}\t{time.time() - start_time}\t{expected_time}\t{' '.join(str(x) for x in classes.values())}")
