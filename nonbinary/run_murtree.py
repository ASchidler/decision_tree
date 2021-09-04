import os
import sys
from sklearn.preprocessing import Binarizer
import nonbinary_instance
from collections import defaultdict

tmp_dir = "/tmp"
murtree_path = "../../../murtree"

instance_path = "nonbinary/instances"
instance_validation_path = "datasets/validate"

fls = list(x for x in os.listdir(instance_path) if x.endswith(".data"))
fls.sort()

target_instance_idx = int(sys.argv[1])

if target_instance_idx > len(fls):
    print(f"Only {len(fls)} files are known.")
    exit(1)

parts = fls[target_instance_idx-1][:-5].split(".")
target_instance = ".".join(parts[:-1])
slice = int(parts[-1])

instance, test_instance, _ = nonbinary_instance.parse(instance_path, target_instance, slice, use_validation=False)

complete_instance = nonbinary_instance.ClassificationInstance()

for c_e in instance:
    complete_instance.add_example(c_e.copy())
for c_e in test_instance:
    complete_instance.add_example(c_e.copy())
complete_instance.finish()

mappings = [dict() for i in range(0, complete_instance.num_features+1)]
bit_lengths = [0 for  i in range(0, complete_instance.num_features+1)]

c_bit = 1

for c_f in range(1, complete_instance.num_features+1):
    bit_lengths = len(complete_instance.domains[c_f])
    for

# Binarize and write out