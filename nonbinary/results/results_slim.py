import os
import nonbinary.tree_parsers as tp
import parser
from collections import defaultdict
import nonbinary.nonbinary_instance as nbi
import pruning

experiment = "i"
flags = set()

ignore = set()

with open("ignore.txt") as ip:
    for _, cl in enumerate(ip):
        if len(cl.strip()) > 0:
            ignore.add(cl.strip())


class TreeData:
    def __init__(self):
        self.nodes = 0
        self.depth = 0
        self.training = 0
        self.test = 0
        self.avg_length = 0

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
            files[file_name][file_fields[3]][file_fields[1]].avg_length = tree.get_avg_length(instance.examples)

            print(f"Parsed {c_file}")
            if file_fields[1] not in original[file_name]:
                tree2 = tp.parse_internal_tree(os.path.join("trees", "unpruned", f"{file_name}.{file_fields[1]}.w.dt"))
                tree2.train(instance)
                original[file_name][file_fields[1]].nodes = tree2.get_nodes()
                original[file_name][file_fields[1]].depth = tree2.get_depth()
                original[file_name][file_fields[1]].training = tree2.get_accuracy(instance.examples)
                original[file_name][file_fields[1]].test = tree2.get_accuracy(instance_test.examples)
                original[file_name][file_fields[1]].avg_length = tree2.get_avg_length(instance.examples)
        else:
            print(f"No tree in {c_file}")

with open(f"results_{experiment}.csv", "w") as outf:
    outf.write("Instance;E;F;C;Nodes;Depth;Train Acc; Test Acc; Avg. Length")
    for c_f in sorted(flags):
        outf.write(f";{c_f} Solved;{c_f} Nodes;{c_f} Depth;{c_f} Train Acc;{c_f} Test Acc;{c_f} Avg. Length")
    outf.write(os.linesep)

    for c_file in sorted(files.keys()):
        c_sizes = sizes[c_file]
        if len(c_sizes) == 0:
            continue
        outf.write(f"{c_file}")
        outf.write(f";{sum(x[0] for x in c_sizes)/len(c_sizes)};{max(x[1] for x in c_sizes)};{max(x[2] for x in c_sizes)}")

        c_data = original[c_file].values()

        sums = TreeData()
        cnt = 0
        for c_data_entry in c_data:
            if c_data_entry.nodes is not None:
                cnt += 1
                sums.nodes += c_data_entry.nodes
                sums.depth += c_data_entry.depth
                sums.training += c_data_entry.training
                sums.test += c_data_entry.test
                sums.avg_length += c_data_entry.avg_length
        for c_s_entry in [sums.nodes, sums.depth, sums.training, sums.test, sums.avg_length]:
            if cnt == 0:
                outf.write(f";{-1}")
            else:
                outf.write(f";{c_s_entry / cnt}")
        print(c_file)
        for c_f in sorted(flags):
            if c_f not in files[c_file]:
                outf.write(";;;;;")
                continue

            c_data = files[c_file][c_f].values()
            sums = TreeData()
            cnt = 0
            for c_data_entry in c_data:
                if c_data_entry.nodes is not None:
                    cnt += 1
                    sums.nodes += c_data_entry.nodes
                    sums.depth += c_data_entry.depth
                    sums.training += c_data_entry.training
                    sums.test += c_data_entry.test
                    sums.avg_length += c_data_entry.avg_length

            outf.write(f";{cnt}")
            for c_s_entry in [sums.nodes, sums.depth, sums.training, sums.test, sums.avg_length]:
                if cnt == 0:
                    outf.write(f";{-1}")
                else:
                    outf.write(f";{c_s_entry / cnt}")
        outf.write(os.linesep)
