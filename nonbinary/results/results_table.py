import sys
from collections import defaultdict

use_pruned = True
use_c45 = False
compare_heur = False

instance_results = defaultdict(list)

# size, depth, acc
indices = [[4, 5, 7], [15, 16, 18]]
if use_pruned:
    indices = [[9, 10, 12], [21, 22, 24]]
    #indices = [[9, 10, 12], [27, 28, 30]]
    indices = [[9, 10, 12], [33, 34, 36]]

ignore = set()
size = {}
with open("ignore.txt") as ip:
    for _, cl in enumerate(ip):
        if len(cl.strip()) > 0:
            ignore.add(cl.strip())

with open("results_d_comp.csv" if (use_c45 or compare_heur) else "results_d_comp_c.csv") as inp_d:
    for i, cl in enumerate(inp_d):
        if i > 0:
            fd = cl.strip().split(";")
            if fd[0] in ignore:
                continue
            size[fd[0]] = float(fd[4])
            instance_results[fd[0]].append([float(fd[x]) for x in indices[0]])
            if not compare_heur:
                instance_results[fd[0]].append([float(fd[x]) for x in indices[1]])

# with open("results_z_comp_c.csv") as inp_d:
#     for i, cl in enumerate(inp_d):
#         if i > 0:
#             fd = cl.strip().split(";")
#             if fd[0] in ignore:
#                 continue
#             instance_results[fd[0]].insert(1, [float(fd[x]) for x in indices[0]])

with open("results_z_comp.csv" if (use_c45 and not compare_heur) else "results_z_comp_c.csv") as inp_d:
    for i, cl in enumerate(inp_d):
        if i > 0:
            fd = cl.strip().split(";")
            if fd[0] in ignore:
                continue
            if not compare_heur:
                instance_results[fd[0]].append([float(fd[x]) for x in indices[1]])
            else:
                instance_results[fd[0]].append([float(fd[x]) for x in indices[0]])

num_results = len(next(iter(instance_results.values())))
bests = [[0, 0, 0] for _ in range(0, num_results)]

print("\\begin{table}")
sys.stdout.write("\\begin{tabular}{l")
for i in range(0, num_results):
    sys.stdout.write("rrr")
print("}")

for ci in sorted(instance_results.keys(), key=lambda x: size[x]):
    sys.stdout.write(ci.replace("_", " "))
    cv = instance_results[ci]
    best_vals = []
    for cf in range(0, 3):
        best_val = min(cv[x][cf] for x in range(0, num_results)) if cf != 2 else max(round(cv[x][cf], 2) for x in range(0, num_results))
        best_vals.append(best_val)

    for i in range(0, len(cv)):
        for cf in range(0, 3):
            if cf == 2:
                cv[i][cf] = round(cv[i][cf], 2)
                strrp = f"{cv[i][cf]:.2f}"
            else:
                strrp = f"{cv[i][cf]:.1f}"

            if best_vals[cf] == cv[i][cf]:
                bests[i][cf] += 1
                sys.stdout.write(f"&\\textbf{{{strrp}}}")
            else:
                sys.stdout.write(f"&{strrp}")
    print("\\\\")

print("\\end{tabular}")
print("\\end{table}")

print(bests)
print(f"/{len(instance_results)}")