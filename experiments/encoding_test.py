import time

import diagram_depth
import tree_depth_encoding
import diagram_encoding
import tree_node_encoding
import sys
import os
import parser
import sat_tools
import aaai_encoding

sample_path = sys.argv[1]
timeout = 1000
memlimit = 2048 * 5

enc_idx = int(sys.argv[2])
slv_idx = int(sys.argv[3])

tmpdir = "./" if len(sys.argv) == 4 else sys.argv[4]
outdir = "./" if len(sys.argv) == 4 else sys.argv[5]

encodings = [
    diagram_encoding.DecisionDiagramEncoding,
    diagram_depth.DiagramDepthEncoding,
    tree_depth_encoding.TreeDepthEncoding,
    tree_node_encoding.TreeEncoding,
    aaai_encoding.AAAIEncoding
]

solvers = [
    sat_tools.MiniSatSolver,
    sat_tools.GlucoseSolver,
    sat_tools.CadicalSolver
]

encoding = encodings[enc_idx]
runner = sat_tools.SatRunner(encoding, solvers[slv_idx](), base_path=tmpdir)

fln_name = os.path.join(outdir, f"results_encodings_{enc_idx}_{slv_idx}.csv")
if not os.path.exists(fln_name):
    with open(fln_name, "w") as out_file:
        out_file.write("project;sample;total;successful;nodes;depth;runtime;accuracy;size")
        out_file.write(os.linesep)

done = {}
#done = {"shuttleM": set(["0.1", "0.05"])}

with open(fln_name, "r+") as out_file:
    for idx, ln in enumerate(out_file):
        if idx > 0:
            cols = ln.split(";")
            if cols[0] not in done:
                done[cols[0]] = set()
            done[cols[0]].add(cols[1])

    for project in os.listdir(sample_path):
        # project directories
        if os.path.isdir(os.path.join(sample_path, project)):
            print(f"Found project {project}")
            sample_paths = []
            for sample in os.listdir(os.path.join(sample_path, project)):
                if os.path.isdir(os.path.join(sample_path, project, sample)):
                    sample_paths.append(os.path.join(sample_path, project, sample))

            sample_paths.sort()
            for sample in sample_paths:
                sample_name = os.path.split(sample)[-1]
                if project in done and sample_name in done[project]:
                    continue
                print(f"  Starting sample {sample_name}")
                cnt_success = 0
                sum_acc = 0
                sum_depth = 0
                sum_node = 0
                sum_time = 0
                enc_size = 0
                cnt = 0
                for in_file in os.listdir(sample):
                    if os.path.isfile(os.path.join(sample, in_file)) and in_file.endswith("training.csv"):
                        cnt += 1
                        print(f"    Starting file {in_file}")
                        new_instance = parser.parse(os.path.join(sample, in_file))
                        test_instance = parser.parse(os.path.join(sample, in_file[0:-1 * len("training.csv")] + "test.csv"))
                        start = time.time()
                        tree, enc_size = runner.run(new_instance, encoding.new_bound(None, new_instance),
                           timeout=timeout, memlimit=memlimit)

                        elapsed = time.time() - start

                        if tree is not None:
                            print(f"     Tree found, Nodes {tree.get_nodes()}, Depth {tree.get_depth()}, Accuracy {tree.get_accuracy(test_instance.examples)}, Time {elapsed}")
                            cnt_success += 1
                            sum_acc += tree.get_accuracy(test_instance.examples)
                            sum_node += tree.get_nodes()
                            sum_depth += tree.get_depth()
                            sum_time += elapsed
                            enc_size += enc_size
                        else:
                            print("      Tree not found in time")
                cnt_divisor = cnt_success if cnt_success > 0 else 1
                out_file.write(f"{project};{sample_name};{cnt};{cnt_success};{sum_node/cnt_divisor};{sum_depth/cnt_divisor};"
                               f"{sum_time/cnt_divisor};{sum_acc/cnt_divisor};{enc_size/cnt_divisor}{os.linesep}")
                out_file.flush()
                print("Finished sample")
            print("Finished project")

