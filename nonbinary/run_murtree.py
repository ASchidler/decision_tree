import os
import sys
from sklearn.preprocessing import Binarizer
import nonbinary_instance

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

