from improve.tree_parsers import parse_iti_tree, parse_weka_tree
import sys
import os
import parser
import improve.improver as imp
import time
import bdd_instance
import re

instance = tree_path = "datasets/trees"
instance_path = "datasets/split"
tmp_dir = "."
is_iti = False

i = 2
while i < len(sys.argv):
    if sys.argv[i] == "-i":
        is_iti = True
    elif sys.argv[i] == "-t":
        tmp_dir = sys.argv[i+1]
        i += 1
    else:
        print(f"Unknown argument {sys.argv[i]}")
    i += 1

fls = list(x for x in os.listdir(instance_path) if x.endswith(".data") and (x.endswith("0.data") or not re.search("[1-4]\.data", x)))
fls.sort()
target_instance_idx = int(sys.argv[1])

if target_instance_idx > len(fls):
    print(f"Only {len(fls)} files are known.")
    exit(1)

target_instance = fls[target_instance_idx-1][:-5]

training_instance = parser.parse(os.path.join(instance_path, target_instance + ".data"), has_header=False)
print(f"{target_instance}: Features {training_instance.num_features}\tExamples {len(training_instance.examples)}\t"
      f"Heuristic {'Weka' if not is_iti else 'ITI'}")

if is_iti:
    tree = parse_iti_tree(os.path.join(tree_path, target_instance+".iti"), training_instance)
else:
    tree = parse_weka_tree(os.path.join(tree_path, target_instance+".tree"), training_instance)

# Run test
done = set()
runner = imp.build_runner(tmp_dir)


def find_leaf():
    q = [tree.root]
    p = {tree.root.id: None}

    while q:
        c_n = q.pop()
        if c_n.is_leaf:
            if c_n.id not in done:
                pth = []
                while c_n:
                    pth.append(c_n)
                    c_n = p[c_n.id]
                return pth
        else:
            p[c_n.left.id] = c_n
            p[c_n.right.id] = c_n
            q.append(c_n.left)
            q.append(c_n.right)

    return None


assigned = imp.assign_samples(tree, training_instance)

for dl in range(3, 20):
    for sl in [10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]:
        done.clear()

        while True:
            c_lf = find_leaf()
            at_limit = False
            if not c_lf:
                break
            c_rt = None

            for idx in range(0, len(c_lf)):
                if imp.depth_from(c_lf[idx]) <= dl:
                    if len(assigned[c_lf[idx].id]) <= sl:
                        c_rt = c_lf[idx]
                    else:
                        break
                else:
                    break

            if c_rt:
                c_d = imp.depth_from(c_rt)
                if c_d == dl and len(assigned[c_rt.id]) <= sl:
                    pass

                if c_d >= 2:
                    new_instance = bdd_instance.BddInstance()
                    for s in assigned[c_rt.id]:
                        new_instance.add_example(training_instance.examples[s].copy())

                    if len(new_instance.examples) > 0:
                        stm = time.time()
                        new_tree, _ = runner.run(new_instance, c_d - 1, u_bound=c_d - 1, timeout=900, suboptimal=True)
                        etm = time.time()

                        if new_tree is None:
                            if (etm - stm) < 899:
                                print(f"{dl}\t{sl}\tNF\t{etm - stm:.2f}\t{c_d}\t{len(new_instance.examples)}")
                                sys.stdout.flush()
                            else:
                                print(f"{dl}\t{sl}\tTO\t{etm - stm:.2f}\t{c_d}\t{len(new_instance.examples)}")
                                finished_limit = True
                                sys.stdout.flush()
                        else:
                            print(f"{dl}\t{sl}\tNT\t{etm - stm:.2f}\t{c_d}\t{len(new_instance.examples)}\t{new_tree.get_depth()}")
                            sys.stdout.flush()
                q = [c_rt]
                while q:
                    c_n = q.pop()
                    done.add(c_n.id)

                    if not c_n.is_leaf:
                        q.extend([c_n.left, c_n.right])

            else:
                print(f"{dl}\t{sl}\tOOB")
                sys.stdout.flush()
                done.add(c_lf[0].id)
