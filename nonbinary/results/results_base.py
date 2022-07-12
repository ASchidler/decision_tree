import math
import os
import sys

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
        self.time = None

files = defaultdict(lambda: defaultdict(lambda: defaultdict(TreeData)))
sizes = defaultdict(list)

for c_file in sorted(os.listdir(os.path.join("trees", experiment))):
    if c_file.find(".") < 0:
        continue
    file_fields = c_file.split(".")
    file_name = file_fields[0]
    flags.add(file_fields[3])
    if c_file.endswith(".info"):
        with open(os.path.join("trees", experiment, c_file)) as info_file:
            files[file_name][file_fields[3]][file_fields[1]].depth_lb = int(info_file.readline().strip())
            info_file.readline()
            time_used = info_file.readline().strip()
            if time_used != "None" and time_used != '':
                files[file_name][file_fields[3]][file_fields[1]].time = float(time_used)
    elif c_file.endswith(".dt"):
        instance, instance_test, _ = nbi.parse("../instances", file_name, int(file_fields[1]))
        tree = tp.parse_internal_tree(os.path.join("trees", experiment, c_file))

        sizes[file_name].append((len(instance.examples), instance.num_features, len(instance.classes), sum(len(x) for x in instance.domains)))

        if tree is not None:
            tree.train(instance)
            files[file_name][file_fields[3]][file_fields[1]].nodes = tree.get_nodes()
            files[file_name][file_fields[3]][file_fields[1]].depth = tree.get_depth()
            files[file_name][file_fields[3]][file_fields[1]].training = tree.get_accuracy(instance.examples)
            files[file_name][file_fields[3]][file_fields[1]].test = tree.get_accuracy(instance_test.examples)

            print(f"Parsed {c_file}")
        else:
            print(f"No tree in {c_file}")

# Get aggregates for encoding

solved = set()
solved_enc = defaultdict(set)
solved_enc_thresh = defaultdict(lambda: defaultdict(set))

thresholds = [60, 300, 600, 3600, 3 * 3600]

for c_f in sorted(flags):
    cnt_unique = 0
    time_common = []
    cnt_total = 0
    var_time_common = 0

    is_size = c_f.find("z") > -1
    # TODO: Fix to exclude categorical tests
    if c_f.find("c") > -1 or is_size or c_f.startswith("1") or c_f.startswith("2") or c_f.startswith("3")  or c_f.startswith("6"):
        continue

    for c_file, c_file_v in files.items():
        if c_f in c_file_v:
            avg_time = 0
            num_slices = 0
            for c_slice, c_slice_data in c_file_v[c_f].items():
                if c_slice_data.time is not None and c_slice_data.time <= 3600.0 * 6:
                    num_slices =+ 1
                    avg_time += c_slice_data.time
                    solved_enc[c_f].add(c_file)
                    solved.add((c_file, c_slice))
                    is_in_none = True
                    is_in_all = True
                    cnt_total += 1
                    for c_f2 in flags:
                        #if c_f2 != c_f and ((c_f2.find("z") > -1) == is_size):
                        if c_f2 != c_f and c_f2.find("z") == -1 and c_f2.find("c") == -1 and not c_f2.startswith("1") and not c_f2.startswith("2") and not c_f2.startswith("3")  and not c_f2.startswith("6"):
                            if c_f2 in c_file_v and c_slice in c_file_v[c_f2] and c_file_v[c_f2][c_slice].depth is not None:
                                is_in_none = False
                            else:
                                is_in_all = False

                    if is_in_none:
                        cnt_unique += 1
                    elif is_in_all:
                        time_common.append(c_slice_data.time)
            if num_slices > 0:
                avg_time /= num_slices
                for c_t in thresholds:
                    if avg_time <= c_t:
                        solved_enc_thresh[c_f][c_t].add(c_file)
    time_avg = sum(time_common) / len(time_common)
    time_sigma = math.sqrt(sum((x - time_avg)**2 for x in time_common) / len(time_common))
    print(f"{c_f} {cnt_total} {cnt_unique} {time_avg} {time_sigma} {len(time_common)}")

print(f"Totally solved: {solved}")

all_instances = set()
for c_f, c_solved in solved_enc.items():
    all_instances.update(c_solved)
    c_unique = set(c_solved)
    for c_of, c_slvd in solved_enc.items():
        if c_of != c_f:
            c_unique -= c_slvd

    ctr = solved_enc_thresh[c_f]
    results_thresh = [str(len(ctr[c_t])) for c_t in thresholds]

    print(f"{c_f} & {len(c_solved)} & {len(c_unique)} "+ "& ".join(results_thresh))

print(f"Totally solved per file: {len(all_instances)}")

# Get aggregates for decision tree types
#
# groups = {"0": {"0", "a", "s"}, "c": {"c"}, "y": {"y"}}
#
# for c_g, c_fs in groups.items():
#     depths = []
#     accuracies = []
#     sizes = []
#     size_accuracies = []
#
#     for c_pfx in ["", "z"]:
#         for c_file, c_file_v in files.items():
#             done = False
#             for c_f in c_fs:
#                 if done:
#                     break
#                 if c_pfx + c_f in c_file_v:
#                     for c_slice, c_slice_data in c_file_v[c_f].items():
#                         if c_slice_data.time is not None and c_slice_data.time <= 3600.0 * 6:
#                             is_in_all = True
#                             for c_g2, c_fs2 in groups.items():
#                                 if c_g2 != c_g:
#                                     is_in_any = False
#                                     for c_f2 in c_fs2:
#                                         if c_pfx + c_f2 in c_file_v and c_slice in c_file_v[c_pfx + c_f2] and c_file_v[c_f2][c_slice].depth is not None:
#                                             is_in_any = True
#                                             break
#                                     if not is_in_any:
#                                         is_in_all = False
#                                         break
#
#                             if is_in_all:
#                                 done = True
#                                 if c_pfx == "":
#                                     depths.append(c_slice_data.depth)
#                                     accuracies.append(c_slice_data.test)
#                                 else:
#                                     sizes.append(c_slice_data.nodes)
#                                     size_accuracies.append(c_slice_data.test)
#
#     print(c_g)
#     for c_metric in [depths, accuracies, sizes, size_accuracies]:
#         metric_avg = sum(c_metric) / len(c_metric)
#         metric_sigma = math.sqrt(sum((x - metric_avg)**2 for x in c_metric) / len(c_metric))
#         print(f"{metric_avg} {metric_sigma}")
#     print("")

with open(f"results_{experiment}.csv", "w") as outf:
    outf.write("Instance;E;F;Values;C")
    for c_f in sorted(flags):
        outf.write(f";{c_f} Solved;{c_f} Depth LB;{c_f} Nodes;{c_f} Depth;{c_f} Train Acc;{c_f} Test Acc;{c_f} Time")
    outf.write(os.linesep)

    for c_file in sorted(files.keys()):
        c_sizes = sizes[c_file]
        if len(c_sizes) == 0:
            continue
        outf.write(f"{c_file}")
        outf.write(f";{sum(x[0] for x in c_sizes)/len(c_sizes)};{max(x[1] for x in c_sizes)};{sum(x[3] for x in c_sizes)/len(c_sizes)};{max(x[2] for x in c_sizes)}")
        print(c_file)
        for c_f in sorted(flags):
            if c_f not in files[c_file]:
                outf.write(";;;;;;;")
                continue

            c_data = files[c_file][c_f].values()
            sums = [0, 0, 0, 0, 0, 0]
            cnt = 0
            time_cnt = 0
            for c_data_entry in c_data:
                if c_data_entry.depth_lb is not None:
                    sums[0] += c_data_entry.depth_lb
                if c_data_entry.nodes is not None:
                    cnt += 1
                    sums[1] += c_data_entry.nodes
                    sums[2] += c_data_entry.depth
                    sums[3] += c_data_entry.training
                    sums[4] += c_data_entry.test
                    if c_data_entry.time is not None:
                        sums[5] += c_data_entry.time
                        time_cnt += 1

            outf.write(f";{cnt}")
            outf.write(f";{sums[0] / len(c_data)}")
            for c_s_entry in range(1, len(sums)-1):
                if cnt == 0:
                    outf.write(f";{-1}")
                else:
                    outf.write(f";{sums[c_s_entry] / cnt}")
            if time_cnt > 0:
                outf.write(f";{sums[-1]/ time_cnt}")
            else:
                outf.write(f";{-1}")

        outf.write(os.linesep)


