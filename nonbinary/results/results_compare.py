import os
import nonbinary.tree_parsers as tp
import parser
from collections import defaultdict
import nonbinary.nonbinary_instance as nbi
import pruning

experiment = "n"
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
        self.avg_length = None

files = defaultdict(lambda: defaultdict(lambda: [defaultdict(TreeData), defaultdict(TreeData)]))
sizes = defaultdict(list)

original = defaultdict(lambda: [defaultdict(TreeData), defaultdict(TreeData)])

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
        tree_p = None
        if os.path.exists(os.path.join("trees", "p", c_file)):
            tree_p = tp.parse_internal_tree(os.path.join("trees", "p", c_file))
        sizes[file_name].append((len(instance.examples), instance.num_features, len(instance.classes)))

        for c_idx, c_t in enumerate([tree, tree_p]):
            if c_t is not None:
                tree.train(instance)
                files[file_name][file_fields[3]][c_idx][file_fields[1]].nodes = c_t.get_nodes()
                files[file_name][file_fields[3]][c_idx][file_fields[1]].depth = c_t.get_depth()
                files[file_name][file_fields[3]][c_idx][file_fields[1]].training = c_t.get_accuracy(instance.examples)
                files[file_name][file_fields[3]][c_idx][file_fields[1]].test = c_t.get_accuracy(instance_test.examples)
                files[file_name][file_fields[3]][c_idx][file_fields[1]].avg_length = c_t.get_avg_length(instance_test.examples)

                print(f"Parsed {c_file}")
                if file_fields[1] not in original[file_name]:
                    if os.path.exists(os.path.join("trees", "unpruned" if c_idx == 0 else "pruned",
                                                                f"{file_name}.{file_fields[1]}.w.dt")):
                        tree2 = tp.parse_internal_tree(os.path.join("trees", "unpruned" if c_idx == 0 else "pruned",
                                                                    f"{file_name}.{file_fields[1]}.w.dt"))
                        tree2.train(instance)
                        original[file_name][c_idx][file_fields[1]].nodes = tree2.get_nodes()
                        original[file_name][c_idx][file_fields[1]].depth = tree2.get_depth()
                        original[file_name][c_idx][file_fields[1]].training = tree2.get_accuracy(instance.examples)
                        original[file_name][c_idx][file_fields[1]].test = tree2.get_accuracy(instance_test.examples)
                        original[file_name][c_idx][file_fields[1]].avg_length = tree2.get_avg_length(instance_test.examples)
            else:
                print(f"No tree in {c_file} {c_idx}")

with open(f"results_{experiment}_comp.csv", "w") as outf:
    outf.write("Instance;E;F;C")
    for c_p in ["U", "P"]:
        outf.write(f";{c_p} Nodes;{c_p} Depth;{c_p} Train Acc;{c_p} Test Acc;{c_p} Avg. Test Length")

    for c_f in sorted(flags):
        for c_p in ["U", "P"]:
            outf.write(f";{c_f} {c_p} Solved;{c_f} {c_p} Nodes;{c_f} {c_p} Depth;{c_f} {c_p} Train Acc;{c_f} {c_p} Test Acc;{c_f} {c_p} Avg. Test Length")

    outf.write(os.linesep)

    for c_file in sorted(files.keys()):
        c_sizes = sizes[c_file]
        if len(c_sizes) == 0:
            continue
        outf.write(f"{c_file}")
        outf.write(f";{sum(x[0] for x in c_sizes)/len(c_sizes)};{max(x[1] for x in c_sizes)};{max(x[2] for x in c_sizes)}")

        for c_idx in [0, 1]:
            c_data = original[c_file][c_idx].values()
            sums = [0, 0, 0, 0, 0]
            cnt = 0
            for c_data_entry in c_data:
                if c_data_entry.nodes is not None:
                    cnt += 1
                    sums[0] += c_data_entry.nodes
                    sums[1] += c_data_entry.depth
                    sums[2] += c_data_entry.training
                    sums[3] += c_data_entry.test
                    sums[4] += c_data_entry.avg_length
            for c_s_entry in range(0, len(sums)):
                if cnt == 0:
                    outf.write(f";{-1}")
                else:
                    outf.write(f";{sums[c_s_entry] / cnt}")
            print(c_file)

        for c_f in sorted(flags):
            for c_idx in [0, 1]:
                if c_f not in files[c_file] or len(files[c_file][c_f]) <= c_idx:
                    outf.write(";;;;;;")
                    continue

                c_data = files[c_file][c_f][c_idx].values()
                sums = [0, 0, 0, 0, 0]
                cnt = 0
                for c_data_entry in c_data:
                    if c_data_entry.nodes is not None:
                        cnt += 1
                        sums[0] += c_data_entry.nodes
                        sums[1] += c_data_entry.depth
                        sums[2] += c_data_entry.training
                        sums[3] += c_data_entry.test
                        sums[4] += c_data_entry.avg_length

                outf.write(f";{cnt}")
                for c_s_entry in range(0, len(sums)):
                    if cnt == 0:
                        outf.write(f";{-1}")
                    else:
                        outf.write(f";{sums[c_s_entry] / cnt}")
        outf.write(os.linesep)
