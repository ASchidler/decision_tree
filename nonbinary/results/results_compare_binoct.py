import os
import sys

ignore = set()
size = {}
with open("ignore.txt") as ip:
    for _, cl in enumerate(ip):
        if len(cl.strip()) > 0:
            ignore.add(cl.strip())


def cmp_vals(vals1, vals2):
    if vals1[2] < vals2[2]:
        return True
    if vals1[2] > vals2[2]:
        return False
    if vals1[0] > vals2[0]:
        return True
    return False

heur_results = {}
heur_indices = [9, 10, 12]
for c_file in ["results_z_comp.csv", "results_z_comp_c.csv"]:
    with open(c_file) as inp:
        for i, cl in enumerate(inp):
            if i > 0:
                cf = cl.split(";")
                vals = [float(cf[x]) for x in heur_indices]
                if cf[0] not in heur_results or cmp_vals(heur_results[cf[0]], vals):
                    heur_results[cf[0]] = vals

slim_indices_all = [[33, 34, 36]]
slim_indices_all = [[27, 28, 30], [21, 22, 24]]
slim_results = {}
for c_file in ["results_z_comp.csv", "results_z_comp_c.csv", "results_d_comp.csv", "results_d_comp_c.csv"]:
    with open(c_file) as inp:
        for i, cl in enumerate(inp):
            if i > 0:
                cf = cl.split(";")
                for slim_indices in slim_indices_all:
                    vals = [float(cf[x]) for x in slim_indices]
                    if cf[0] not in slim_results or cmp_vals(slim_results[cf[0]], vals):
                        slim_results[cf[0]] = vals

binoct_results = {}
with open("results_binoct.csv") as inp:
    for i, cl in enumerate(inp):
        if i > 0:
            cf = cl.strip().split(";")
            for cx in range(0, 6):
                vals = [float(cf[x]) for x in [1 + cx * 5, 1 + cx * 5 + 1, 1 + cx * 5 + 3]]
                if cf[0] not in binoct_results or cmp_vals(binoct_results[cf[0]], vals):
                    binoct_results[cf[0]] = vals

instances = (binoct_results.keys() & heur_results.keys() & slim_results.keys()) - ignore

print("Instance & s & d & a & s & d & a & s & d & a \\\\"+os.linesep)
for c_instance in sorted(instances, key=lambda x: heur_results[x][0]): # key=lambda x: x.upper()):
    sys.stdout.write(c_instance.replace("_", " "))
    bests = []
    sources = [heur_results[c_instance], slim_results[c_instance], binoct_results[c_instance]]
    for cv in range(0, 3):
        if cv != 2:
            for c_source in sources:
                c_source[cv] = round(c_source[cv], 1)
            bests.append(min(x[cv] for x in sources))
        else:
            for c_source in sources:
                c_source[cv] = round(c_source[cv], 2)
            bests.append(max(x[cv] for x in sources))

    for c_source in sources:
        for i, c_v in enumerate(c_source):
            if c_v == bests[i]:
                sys.stdout.write(f"&\\textbf{{{c_v}}}")
            else:
                sys.stdout.write(f"&{c_v}")
    sys.stdout.write("\\\\"+os.linesep)
