import sys
from collections import defaultdict

use_pruned = False

instance_results = defaultdict(list)

# size, depth, acc
indices = [[4, 5, 7], [15, 16, 18]]
if use_pruned:
    indices = [[9, 10, 12], [21, 22, 24]]

with open("results_d_comp.csv") as inp_d:
    for i, cl in enumerate(inp_d):
        if i > 0:
            fd = cl.strip().split(";")
            instance_results[fd[0]].append([float(fd[x]) for x in indices[0]])
            instance_results[fd[0]].append([float(fd[x]) for x in indices[1]])

with open("results_z_comp.csv") as inp_d:
    for i, cl in enumerate(inp_d):
        if i > 0:
            fd = cl.strip().split(";")
            instance_results[fd[0]].append([float(fd[x]) for x in indices[1]])

bests = [[0, 0, 0] for _ in range(0, 3)]

print("\\begin{table}")
sys.stdout.write("\\begin{tabular}{l")
for i in range(0, 3):
    sys.stdout.write("rrr")
print("}")

for ci in sorted(instance_results.keys()):
    sys.stdout.write(ci.replace("_", ""))
    cv = instance_results[ci]
    best_vals = []
    for cf in range(0, 3):
        best_val = min(cv[x][cf] for x in range(0, 3)) if cf != 2 else max(round(cv[x][cf], 2) for x in range(0, 3))
        best_vals.append(best_val)

    for i in range(0, len(cv)):
        for cf in range(0, 3):
            if cf == 2:
                cv[i][cf] = round(cv[i][cf], 2)
            if best_vals[cf] == cv[i][cf]:
                bests[i][cf] += 1
                sys.stdout.write(f"&\\textbf{{{cv[i][cf]}}}")
            else:
                sys.stdout.write(f"&{cv[i][cf]}")
    print("\\\\")

print("\\end{tabular}")
print("\\end{table}")

for i in range(0, 3):
    print(bests)
