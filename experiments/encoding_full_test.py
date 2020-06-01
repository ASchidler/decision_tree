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
import bdd_instance

sample_path = sys.argv[1]
timeout = 1000
memlimit = 2048 * 5

enc_idx = int(sys.argv[2])
slv_idx = int(sys.argv[3])

tmpdir = "./" if len(sys.argv) < 6 else sys.argv[4]
outdir = "./" if len(sys.argv) < 6 else sys.argv[5]

encodings = [
    diagram_encoding.DecisionDiagramEncoding,
    diagram_depth.DiagramDepthEncoding,
    tree_depth_encoding.TreeDepthEncoding,
    tree_node_encoding.TreeEncoding,
    aaai_encoding.AAAIEncoding
]

encoding = encodings[enc_idx]

solvers = [
    sat_tools.MiniSatSolver,
    sat_tools.GlucoseSolver,
    sat_tools.CadicalSolver
]

keys = {}
for fl in os.listdir("."):
    full_name = f"./{fl}"
    if os.path.isfile(full_name) and fl.startswith("keys_"):
        with open(full_name) as key_file:
            for ln in key_file:
                cols = ln.split(";")
                key = cols[1].split(",")[0:-1]

                if cols[0] not in keys or len(keys[cols[0]]) > len(key):
                    keys[cols[0]] = key

        print(f"Processed {fl}")
keys = {k: [int(cv) for cv in v] for k, v in keys.items()}

runner = sat_tools.SatRunner(encoding, solvers[slv_idx](), base_path=tmpdir)

fln_name = os.path.join(outdir, f"results_encoding_full_{enc_idx}_{os.path.split(os.path.normpath(sample_path))[-1]}.csv")
if not os.path.exists(fln_name):
    with open(fln_name, "w") as out_file:
        out_file.write("project;nodes;depth;runtime;size;acc")
        out_file.write(os.linesep)

with open(fln_name, "r+") as out_file:
    done = set()

    for _, cln in enumerate(out_file):
        done.add(cln.split(";")[0])

    samples = list(os.listdir(sample_path))
    samples.sort(key=lambda x: os.path.getsize(os.path.join(sample_path, x)))

    for project in samples:
        if project.endswith(".csv") and not project.endswith("_test.csv"):
            if project.endswith("_training.csv"):
                project_name = project[0:-1 * len("_training.csv")]
                test_instance = project_name + "_test.csv"
            else:
                project_name = project[0:-4]
                test_instance = project
            if project_name in done:
                continue

            print(f"Starting {project_name}")
            new_instance = parser.parse(os.path.join(sample_path, project))
            bdd_instance.reduce(new_instance, min_key=keys[project])
            test_instance = parser.parse(os.path.join(sample_path, test_instance))

            start = time.time()

            tree, enc_size = runner.run(new_instance, encoding.new_bound(None, new_instance),
                                        timeout=timeout, memlimit=memlimit)

            elapsed = time.time() - start

            if tree is not None:
                new_instance.unreduce_instance(tree)
                print(
                    f"Tree found, Nodes {tree.get_nodes()}, Depth {tree.get_depth()}, Time {elapsed}")
                out_file.write(f"{project_name};{tree.get_nodes()};{tree.get_depth()};{elapsed};{enc_size};{tree.get_accuracy(test_instance.examples)}{os.linesep}")
            else:
                print(f"Tree not found")
                out_file.write(f"{project_name};{-1};{-1};{elapsed};{enc_size};{-1}{os.linesep}")
            out_file.flush()
