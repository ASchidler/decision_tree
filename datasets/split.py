import os
import sys
import shutil
import parser
import random
import math

for r, d, f in os.walk("binary"):
    for fn in f:
        if not fn.endswith(".data"):
            continue
        print(f"Processing {fn}")

        if os.path.exists(os.path.join(r, fn[:-4] + "test")):
            shutil.copy(os.path.join(r, fn), os.path.join("split", fn))
            shutil.copy(os.path.join(r, fn[:-4] + "test"), os.path.join("split", fn[:-4] + "test"))
            shutil.copy(os.path.join(r, fn[:-4] + "names"), os.path.join("split", fn[:-4] + "names"))
            continue

        inst = parser.parse(os.path.join(r, fn), has_header=False)

        p_ex = []
        n_ex = []

        for c_ex in inst.examples:
            if c_ex.cls:
                p_ex.append(c_ex)
            else:
                n_ex.append(c_ex)

        # Shuffle
        for i in range(0, len(p_ex)):
            n_idx = random.randint(i, len(p_ex) - 1)
            p_ex[i], p_ex[n_idx] = p_ex[n_idx], p_ex[i]
        for i in range(0, len(n_ex)):
            n_idx = random.randint(i, len(n_ex) - 1)
            n_ex[i], n_ex[n_idx] = n_ex[n_idx], n_ex[i]

        # Create 5 different groups
        remaining = 5
        groups = [[] for _ in range(0, remaining)]

        while remaining > 0:
            t_p = math.ceil(len(p_ex) / remaining)
            t_n = math.floor(len(n_ex) / remaining)

            for _ in range(0, t_p):
                groups[remaining-1].append(p_ex.pop())
            for _ in range(0, t_n):
                groups[remaining-1].append(n_ex.pop())

            remaining -= 1

        # Create instances
        for i in range(0, len(groups)):
            new_fn = f"{fn[:-5]}-{i}"
            shutil.copy(os.path.join(r, fn[:-4] + "names"), os.path.join("split", new_fn + ".names"))

            with open(os.path.join("split", new_fn + ".test"), "w") as outp:
                for c_ex in groups[i]:
                    for cf in c_ex.features[1:]:
                        outp.write(f"{1 if cf else 0},")
                    outp.write(f"{1 if c_ex.cls else 0}{os.linesep}")

            with open(os.path.join("split", new_fn + ".data"), "w") as outp:
                for j in range(0, len(groups)):
                    if i == j:
                        continue

                    for c_ex in groups[j]:
                        for cf in c_ex.features[1:]:
                            outp.write(f"{1 if cf else 0},")
                        outp.write(f"{1 if c_ex.cls else 0}{os.linesep}")
