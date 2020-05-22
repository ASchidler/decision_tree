import os
import sys
import random

in_path = sys.argv[1]
out_path = sys.argv[2]

ratio = 0.2

for r, d, f in os.walk(in_path):
    for fl in f:
        if fl.endswith(".csv"):
            training_name = fl[0:-4] + "_training.csv"
            test_name = fl[0:-4] + "_test.csv"
            with open(os.path.join(r, fl), "r") as c_file:
                # read file
                lines = [x for _, x in enumerate(c_file)]

            # Pick ratio many random lines
            cnt = len(lines) - 1
            target = cnt * ratio
            indices = set()
            while len(indices) <= target:
                indices.add(random.randint(1, len(lines) - 1))

            with open(os.path.join(out_path, test_name), "w") as test_file:
                test_file.write(lines[0])
                for cidx in indices:
                    test_file.write(lines[cidx])

            with open(os.path.join(out_path, training_name), "w") as train_file:
                train_file.write(lines[0])

                for i, ln in enumerate(lines):
                    if i > 0 and i not in indices:
                        train_file.write(ln)


