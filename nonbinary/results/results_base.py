import os
import nonbinary.tree_parsers as tp
import parser
from collections import defaultdict
import nonbinary.nonbinary_instance as nbi
import pruning

experiment = "e"
flags = set()


class TreeData:
    def __init__(self):
        self.nodes = None
        self.depth = None
        self.training = None
        self.test = None
        self.depth_lb = None

files = defaultdict(lambda: defaultdict(lambda: defaultdict(TreeData)))
sizes = defaultdict(list)

for c_file in sorted(os.listdir("trees")):
    file_fields = c_file.split(".")
    file_name = file_fields[0]
    flags.add(file_fields[3])
    if c_file.endswith(".info"):
        with open(os.path.join("trees", c_file)) as info_file:
            files[file_name][file_fields[3]][file_fields[1]].depth_lb = int(info_file.readline().strip())
    elif c_file.endswith(".dt"):
        instance, instance_test, _ = nbi.parse("../instances", file_name, int(file_fields[1]))
        tree = tp.parse_internal_tree(os.path.join("trees", c_file))

        sizes[file_name].append((len(instance.examples), instance.num_features, len(instance.classes)))

        if tree is not None:
            files[file_name][file_fields[3]][file_fields[1]].nodes = tree.get_nodes()
            files[file_name][file_fields[3]][file_fields[1]].depth = tree.get_depth()
            files[file_name][file_fields[3]][file_fields[1]].training = tree.get_accuracy(instance.examples)
            files[file_name][file_fields[3]][file_fields[1]].test = tree.get_accuracy(instance_test.examples)

            print(f"Parsed {c_file}")
        else:
            print(f"No tree in {c_file}")

with open(f"results_{experiment}.csv", "w") as outf:
    outf.write("Instance;E;F;C")
    for c_f in sorted(flags):
        outf.write(f";{c_f} Solved;{c_f} Depth LB;{c_f} Nodes;{c_f} Depth;{c_f} Train Acc;{c_f} Test Acc")
    outf.write(os.linesep)

    for c_file in sorted(files.keys()):
        c_sizes = sizes[c_file]
        if len(c_sizes) == 0:
            continue
        outf.write(f"{c_file}")
        outf.write(f";{sum(x[0] for x in c_sizes)/len(c_sizes)};{max(x[1] for x in c_sizes)};{max(x[2] for x in c_sizes)}")
        print(c_file)
        for c_f in sorted(flags):
            if c_f not in files[c_file]:
                outf.write(";;;;;")
                continue

            c_data = files[c_file][c_f].values()
            sums = [0, 0, 0, 0, 0]
            cnt = 0
            for c_data_entry in c_data:
                sums[0] += c_data_entry.depth_lb
                if c_data_entry.nodes is not None:
                    cnt += 1
                    sums[1] += c_data_entry.nodes
                    sums[2] += c_data_entry.depth
                    sums[3] += c_data_entry.training
                    sums[4] += c_data_entry.test

            outf.write(f";{cnt}")
            outf.write(f";{sums[0] / len(c_data)}")
            for c_s_entry in range(1, len(sums)):
                if cnt == 0:
                    outf.write(f";{-1}")
                else:
                    outf.write(f";{sums[c_s_entry] / cnt}")
        outf.write(os.linesep)
