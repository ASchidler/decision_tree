import os
import nonbinary.tree_parsers as tp
import parser
from collections import defaultdict
import nonbinary.nonbinary_instance as nbi
import pruning

experiment = "u"
flags = set()

ignore = set()

with open("ignore.txt") as ip:
    for _, cl in enumerate(ip):
        if len(cl.strip()) > 0:
            ignore.add(cl.strip())


class TreeData:
    def __init__(self):
        self.nodes = None
        self.depth = None
        self.training = None
        self.test = None

files = defaultdict(lambda: defaultdict(lambda: defaultdict(TreeData)))
sizes = defaultdict(list)

original = defaultdict(lambda: defaultdict(TreeData))

for c_file in sorted(os.listdir(os.path.join("trees", experiment))):
    if c_file.find(".") < 0:
        continue
    file_fields = c_file.split(".")
    file_name = file_fields[0]

    if file_name in ignore:
        continue

    flags.add(file_fields[3])
    if c_file.endswith(".dt"):
        instance, instance_test, _ = nbi.parse("../instances", file_name, int(file_fields[1]))
        tree = tp.parse_internal_tree(os.path.join("trees", experiment, c_file))

        sizes[file_name].append((len(instance.examples), instance.num_features, len(instance.classes)))

        if tree is not None:
            tree.train(instance)
            files[file_name][file_fields[3]][file_fields[1]].nodes = tree.get_nodes()
            files[file_name][file_fields[3]][file_fields[1]].depth = tree.get_depth()
            files[file_name][file_fields[3]][file_fields[1]].training = tree.get_accuracy(instance.examples)
            files[file_name][file_fields[3]][file_fields[1]].test = tree.get_accuracy(instance_test.examples)

            print(f"Parsed {c_file}")
            if file_fields[1] not in original[file_name]:
                tree2 = tp.parse_internal_tree(os.path.join("trees", "unpruned", f"{file_name}.{file_fields[1]}.w.dt"))
                tree2.train(instance)
                original[file_name][file_fields[1]].nodes = tree2.get_nodes()
                original[file_name][file_fields[1]].depth = tree2.get_depth()
                original[file_name][file_fields[1]].training = tree2.get_accuracy(instance.examples)
                original[file_name][file_fields[1]].test = tree2.get_accuracy(instance_test.examples)
        else:
            print(f"No tree in {c_file}")

with open(f"results_{experiment}.csv", "w") as outf:
    outf.write("Instance;E;F;C;Nodes;Depth;Train Acc; Test Acc")
    for c_f in sorted(flags):
        outf.write(f";{c_f} Solved;{c_f} Nodes;{c_f} Depth;{c_f} Train Acc;{c_f} Test Acc")
    outf.write(os.linesep)

    for c_file in sorted(files.keys()):
        c_sizes = sizes[c_file]
        if len(c_sizes) == 0:
            continue
        outf.write(f"{c_file}")
        outf.write(f";{sum(x[0] for x in c_sizes)/len(c_sizes)};{max(x[1] for x in c_sizes)};{max(x[2] for x in c_sizes)}")

        c_data = original[c_file].values()
        sums = [0, 0, 0, 0]
        cnt = 0
        for c_data_entry in c_data:
            if c_data_entry.nodes is not None:
                cnt += 1
                sums[0] += c_data_entry.nodes
                sums[1] += c_data_entry.depth
                sums[2] += c_data_entry.training
                sums[3] += c_data_entry.test
        for c_s_entry in range(0, len(sums)):
            if cnt == 0:
                outf.write(f";{-1}")
            else:
                outf.write(f";{sums[c_s_entry] / cnt}")
        print(c_file)
        for c_f in sorted(flags):
            if c_f not in files[c_file]:
                outf.write(";;;;;")
                continue

            c_data = files[c_file][c_f].values()
            sums = [0, 0, 0, 0]
            cnt = 0
            for c_data_entry in c_data:
                if c_data_entry.nodes is not None:
                    cnt += 1
                    sums[0] += c_data_entry.nodes
                    sums[1] += c_data_entry.depth
                    sums[2] += c_data_entry.training
                    sums[3] += c_data_entry.test

            outf.write(f";{cnt}")
            for c_s_entry in range(0, len(sums)):
                if cnt == 0:
                    outf.write(f";{-1}")
                else:
                    outf.write(f";{sums[c_s_entry] / cnt}")
        outf.write(os.linesep)
