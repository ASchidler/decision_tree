import os
from collections import defaultdict
from nonbinary.nonbinary_instance import parse
from nonbinary.tree_parsers import parse_internal_tree

pth = "nonbinary/results/trees/binoct"
instance_pth = "nonbinary/instances"
outp_path = "nonbinary/results/results_binoct.csv"
max_depth = 6
fls = os.listdir(instance_pth)

results = defaultdict(lambda: defaultdict(list))

for c_fl in fls:
    if c_fl.endswith(".data") and os.path.exists(os.path.join(pth, c_fl[:-4] +"2.dt")):
        print(f"Parsing {c_fl}")
        instance, instance_test, instance_validation = parse(instance_pth, c_fl.split(".")[0], int(c_fl.split(".")[1]))
        for dp in range(2, max_depth+1):
            tree = parse_internal_tree(os.path.join(pth, c_fl[:-4] +f"{dp}.dt"))
            results[c_fl.split(".")[0]][dp].append((tree.get_nodes(), tree.get_depth(), tree.get_accuracy(instance.examples),
                tree.get_accuracy(instance_test.examples), tree.get_avg_length(instance_test.examples)))

with open(outp_path, "w") as outp:
    outp.write("Instance")
    for i in range(2, max_depth + 2):
        nm = str(i) if i <= max_depth else 'VB'
        outp.write(f";{nm} Nodes;{nm} Depth;{nm} Train;{nm} Test;{nm} Avg. Len")
    outp.write(os.linesep)

    for c_fl in sorted(results.keys()):
        c_best = None
        outp.write(f"{c_fl}")
        for i in range(2, max_depth+1):
            c_sums = [0, 0, 0, 0, 0]
            for c_result in results[c_fl][i]:
                for cidx in range(0, len(c_sums)):
                    c_sums[cidx] += c_result[cidx]
            c_sums = [x / len(results[c_fl][i]) for x in c_sums]
            if c_best is None or c_best[3] < c_sums[3]:
                c_best = c_sums
            for cidx in range(0, len(c_sums)):
                outp.write(f";{c_sums[cidx]}")
        for cidx in range(0, len(c_best)):
            outp.write(f";{c_best[cidx]}")
        outp.write(os.linesep)



