import os
import sys
import parser
import bdd_instance

in_path = sys.argv[1]
out_path = sys.argv[2]

keys = {}

for fl in os.listdir("."):
    full_name = f"./{fl}"
    if os.path.isfile(full_name) and fl.startswith("keys_"):
        with open(full_name) as key_file:
            for ln in key_file:
                cols = ln.split(";")
                key = cols[1].split(",")[0:-1]

                if cols[0] not in keys or len(keys[cols[0]]) > len(key):
                    keys[cols[0]] = key

        print(f"Processed {fl}")

keys = {k: [int(cv) for cv in v] for k, v in keys.items()}

for fl in os.listdir(in_path):
    full_name = os.path.join(in_path, fl)
    if os.path.isfile(full_name) and full_name.endswith(".csv"):
        fn = fl
        if fn.endswith("_training.csv"):
            fn = fn[0:-1*len("_training.csv")] + ".csv"
        elif fn.endswith("_test.csv"):
            fn = fn[0:-1*len("_test.csv")] + ".csv"

        if fn not in keys:
            print(f"No key found for {fn} ({fl})")
            continue

        inst = parser.parse(full_name)
        bdd_instance.reduce(inst, min_key=keys[fn])

        new_fn = fn[0:-4]
        if new_fn.endswith("-un"):
            new_fn = new_fn[0:-3]

        new_fn = fl.replace(new_fn, new_fn + "-red")

        with open(os.path.join(out_path, new_fn), "w") as out_file:
            for f in range(0, inst.num_features):
                out_file.write(f"a{f},")
            out_file.write(f"c{os.linesep}")

            for e in inst.examples:
                for f in range(1, inst.num_features+1):
                    out_file.write(f"{e.features[f]},")
                out_file.write(f"{e.cls}{os.linesep}")

