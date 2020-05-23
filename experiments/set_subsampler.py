import os
import sys
import random
import parser

in_path = sys.argv[1]
out_path = sys.argv[2]

test_ratio = 0.25
ratios = {
    "appendicitis": [0.3, 0.4],
    "australian": [0.05, 0.1],
    "backache": [0.3, 0.4],
    "cancer": [0.1, 0.2, 0.25],
    "car": [0.05, 0.1],
    "cleve": [0.1],
    "colic": [0.05, 0.1],
    "shuttleM": [0.05, 0.1]
}

# Store target ratios
# sample 0.25 test and target ratios
# => One training and one test set
# Capture encoding size as well

for fl in os.listdir(in_path):
    if os.path.isfile(os.path.join(in_path, fl)) and fl.endswith(".csv"):
        target = None
        # Find ratios
        for cv in ratios.keys():
            if fl.startswith(cv):
                target = cv
                break

        # File should not be included
        if target is None:
            continue

        target_ratios = ratios[target]

        new_path = os.path.join(out_path, target)
        if not os.path.exists(new_path):
            os.mkdir(new_path)

        inst = parser.parse(os.path.join(in_path, fl))
        indices = list(range(0, len(inst.examples)))
        test_count = int(len(indices) * test_ratio)

        for c_ratio in target_ratios:
            training_count = int(len(indices) * c_ratio)
            c_path = os.path.join(new_path, f"{c_ratio}")
            if not os.path.exists(c_path):
                os.mkdir(c_path)

            for c_subsample in range(1, 21):
                # Find random arrangement of indices
                for i in range(0, len(indices)):
                    new_idx = random.randint(i, len(indices)-1)
                    indices[i], indices[new_idx] = indices[new_idx], indices[i]

                def write_file(tf, s, e):
                    for cf in range(0, inst.num_features):
                        tf.write(f"att{cf},")
                    tf.write(f"c{os.linesep}")

                    for ci in range(s, e+1):
                        idx = indices[ci]
                        for cf in range(0, inst.num_features):
                            tf.write(f"{inst.examples[idx].features[cf]},")
                        tf.write(f"{inst.examples[idx].cls}{os.linesep}")

                # Write test set
                with open(os.path.join(c_path, f"{target}_{c_ratio}_{c_subsample}_test.csv"), "w") as test_file:
                    write_file(test_file, 0, test_count)
                # Write training set
                with open(os.path.join(c_path, f"{target}_{c_ratio}_{c_subsample}_training.csv"), "w") as test_file:
                    write_file(test_file, test_count+1, test_count + training_count)



