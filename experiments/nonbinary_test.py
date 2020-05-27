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

slv_idx = int(sys.argv[2])

tmpdir = "./" if len(sys.argv) == 3 else sys.argv[3]
outdir = "./" if len(sys.argv) == 3 else sys.argv[4]

encoding = tree_depth_encoding.TreeDepthEncoding

solvers = [
    sat_tools.MiniSatSolver,
    sat_tools.GlucoseSolver,
    sat_tools.CadicalSolver
]

runner = sat_tools.SatRunner(encoding, solvers[slv_idx](), base_path=tmpdir)

fln_name = os.path.join(outdir, f"results_nonbinary_{slv_idx}.csv")
if not os.path.exists(fln_name):
    with open(fln_name, "w") as out_file:
        out_file.write("project;nodes;depth;runtime;size")
        out_file.write(os.linesep)

with open(fln_name, "r+") as out_file:
    done = set()
    for _, cln in enumerate(out_file):
        done.add(cln.split(";")[0])

    for project in os.listdir(sample_path):
        if project.endswith(".csv"):
            project_name = project[0:-4]
            if project_name in done:
                continue

            print(f"Starting {project_name}")
            new_instance = parser.parse_nonbinary(os.path.join(sample_path, project))
            start = time.time()

            tree, enc_size = runner.run(new_instance, encoding.new_bound(None, new_instance),
                                        timeout=timeout, memlimit=memlimit)

            elapsed = time.time() - start

            if tree is not None:
                print(
                    f"Tree found, Nodes {tree.get_nodes()}, Depth {tree.get_depth()}, Time {elapsed}")
                out_file.write(f"{project_name};{tree.get_nodes()};{tree.get_depth()};{elapsed};{enc_size}{os.linesep}")
            else:
                print(f"Tree not found")
                out_file.write(f"{project_name};{-1};{-1};{elapsed};{enc_size}{os.linesep}")
            out_file.flush()
