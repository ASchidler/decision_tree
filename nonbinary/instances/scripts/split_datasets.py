from sklearn.model_selection import StratifiedKFold
import os
import shutil

base_path = "nonbinary/instances"

header_names = None

for fl in sorted(os.listdir(base_path)):
    if fl.endswith(".data"):
        fl_path = os.path.join(base_path, fl)
        suffix = fl[:-5].split(".")
        # Is already a slice
        if len(suffix) > 1:
            continue

        # With a test set, create 4 extra slices, else 5
        slices = 4 if os.path.exists(fl_path[:-4] + "test") else 5

        # Load file
        x = []
        y = []

        with open(os.path.join(base_path, fl)) as inp:
            for cl in inp:
                fields = cl.split(",")
                x.append(fields[:-1])
                y.append(fields[-1].strip())

        x_slices = list(StratifiedKFold(n_splits=slices).split(x, y))

        for i in range(1, slices+1):
            shutil.copy(fl_path[:-4] + "names", fl_path[:-4] + f"{i}.names")
            with open(fl_path[:-4] + f"{i}.data", "w") as outp:
                for c_i in x_slices[i-1][0]:
                    outp.write(",".join(x[c_i]) + f"," + y[c_i] + os.linesep)

        os.remove(fl_path)
