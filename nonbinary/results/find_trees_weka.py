import os
import shutil
import subprocess
import re
import sys
from sys import maxsize
import improve.tree_parsers as tp
import parser
from nonbinary.nonbinary_instance import ClassificationInstance, Example, parse
from collections import deque
from nonbinary.decision_tree import DecisionTree, DecisionTreeLeaf, DecisionTreeNode
from decimal import Decimal, InvalidOperation
import resource

resource.setrlimit(resource.RLIMIT_AS, (23 * 1024 * 1024 * 1024 // 2, 12 * 1024 * 1024 * 1024))
use_validation = False
pruning = 1  # 0 is no pruning
categorical = False
sys.setrecursionlimit(5000)

pth = "nonbinary/instances"
weka_path = os.path.join(os.path.expanduser("~"), "Downloads/weka-3-8-5-azul-zulu-linux/weka-3-8-5")
jre_path = os.path.join(weka_path, "jre/zulu11.43.55-ca-fx-jre11.0.9.1-linux_x64/bin/java")
#-cp ./weka.jar

# Only for non pruned trees
parameters = ["-U", "-J", "-M", "0", "-B"]

class WekaNode:
    def __init__(self, feat=None, threshold=None, is_cat=None, cls=None):
        self.cls = cls
        self.feat = feat
        self.threshold = [threshold]
        self.children = []
        self.is_cat = is_cat


def parse_weka_tree(lines):
    wtree = DecisionTree()
    # Single leaf tree, edge case
    if lines[0].strip().startswith(":"):
        pos = lines[0].index(":")
        pos2 = lines[0].index(" ", pos+2)
        cls = lines[0][pos + 2:pos2]
        c_leaf = DecisionTreeLeaf(cls, 1, wtree)
        wtree.set_root_leaf(c_leaf)
        return wtree

    root = None
    c_id = 1
    l_depth = -1
    stack = []
    for ll in lines:
        depth = 0
        for cc in ll:
            if cc == " " or cc == "|":
                depth += 1
            else:
                c_line = ll[depth:].strip()
                while stack and depth < l_depth:
                    stack.pop()
                    l_depth -= 4

                cp = None if not stack else stack[-1]
                if not c_line.startswith("att"):
                    print(f"Parser error, line should start with att, starts with {c_line}.")
                    exit(1)

                t_str = c_line.split(" ")[2].replace(":", "")  # remove the : at the end
                try:
                    threshold = int(t_str)
                except ValueError:
                    try:
                        threshold = Decimal(t_str)
                    except InvalidOperation:
                        threshold = t_str

                if depth > l_depth:
                    feature = int(c_line[3:c_line.find(" ")])

                    is_cat = c_line.split(" ")[1].strip() == "="
                    if cp is not None:
                        node = WekaNode(feat=feature, threshold=threshold, is_cat=is_cat)
                        cp.children.append(node)
                    else:
                        root = WekaNode(feat=feature, threshold=threshold, is_cat=is_cat)
                        node = root
                    stack.append(node)
                    c_id += 1
                    cp = node
                else:
                    cp.threshold.append(threshold)

                if c_line.find(":") > -1:
                    pos = c_line.index(":")
                    pos2 = c_line.index(" ", pos+2)
                    cls = c_line[pos + 2:pos2]
                    cp.children.append(WekaNode(cls=cls))
                    c_id += 1

                l_depth = depth
                break

    def construct_tree(cn, parent, cp):
            if cn.cls is not None:
                if parent is None:
                    wtree.set_root_leaf(cn.cls)
                else:
                    wtree.add_leaf(cn.cls, parent, cp)
            else:
                cn.threshold.reverse()
                cn.children.reverse()

                if parent is None:
                    n_n = wtree.set_root(cn.feat, cn.threshold.pop(), cn.is_cat)
                else:
                    n_n = wtree.add_node(cn.feat, cn.threshold.pop(), parent, cp, cn.is_cat)
                construct_tree(cn.children.pop(), n_n.id, True)

                if len(cn.threshold) == 1:
                    construct_tree(cn.children.pop(), n_n.id, False)
                else:
                    construct_tree(cn, n_n.id, False)

    construct_tree(root, None, None)
    return wtree


def get_tree(instance_path, params):
    """Calls WEKA"""
    process = subprocess.Popen([
        jre_path, "-cp", os.path.join(weka_path, "weka.jar"),
        "weka.classifiers.trees.J48",
        "-t", instance_path, "-no-cv",
        *params],
        cwd=weka_path,
        stderr=subprocess.PIPE,#open(os.devnull, 'w'),
        stdout=subprocess.PIPE
        )

    output, oute = process.communicate()
    output = output.decode('ascii')

    mt = re.search("J48 u?n?pruned tree[^\-]*[\-]*(.*)Number of Leaves", output, re.DOTALL)


    return mt.group(1).strip().splitlines()


fls = {".".join(x.split(".")[:-2]) for x in list(os.listdir(pth)) if x.endswith(".data")}
fls = sorted(fls, key=lambda x:os.path.getsize(os.path.join(pth, f"{x}.1.data")))

for fl in fls:
    print(f"{fl}")
    for c_slice in range(1, 6):
        print(f"{c_slice}")

        if categorical:
            output_path = f"nonbinary/results/trees/categorical/{fl}.{c_slice}.w.dt"
        elif pruning == 0:
            fld = "unpruned" if not use_validation else "validation"
            output_path = f"nonbinary/results/trees/{fld}/{fl}.{c_slice}.w.dt"
        else:
            output_path = f"nonbinary/results/trees/pruned/{fl}.{c_slice}.w.dt"

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(os.path.getsize(output_path))
            continue

        try:
            instance, instance_test, instance_validation = parse(pth, fl, c_slice, use_validation=use_validation or pruning == 1)
        except FileNotFoundError:
            # Invalid slice for instances with test set.
            continue

        f_map = [{}]
        if categorical:
            original_instance = instance
            for c_f in range(1, instance.num_features+1):
                c_id = 1
                c_f_map = {}
                f_map.append(c_f_map)
                for c_v in instance.domains[c_f]:
                    c_f_map[c_v] = c_id
                    c_id += 1

            instance = ClassificationInstance()
            for c_e in original_instance.examples:
                instance.add_example(Example(instance, ["?" if x == "?" else f_map[c_f+1][x] for c_f, x in enumerate(c_e.features[1:])], c_e.cls))

        instance.export_c45("/tmp/weka_instance.data", categorical=categorical)

        if pruning == 0:
            tree = parse_weka_tree(get_tree("/tmp/weka_instance.data", parameters))
            if categorical:
                instance = original_instance
            if categorical:
                r_f_map = [{} for _ in range(0, original_instance.num_features + 1)]
                for c_f, c_entry in enumerate(f_map):
                    for c_v, c_idx in c_entry.items():
                        r_f_map[c_f][c_idx] = c_v
                def fix_c_node(c_node):
                    if not c_node.is_leaf:
                        c_node.threshold = r_f_map[c_node.feature][c_node.threshold]
                        fix_c_node(c_node.left)
                        fix_c_node(c_node.right)
                fix_c_node(tree.root)

            print(f"{tree.get_accuracy(instance.examples)}")
            with open(output_path, "w") as outp:
                outp.write(tree.as_string())
        else:
            def get_accuracy(c_val, m_val):
                c_tree = parse_weka_tree(get_tree("/tmp/weka_instance.data", ["-B", "-C", f"{c_val}", "-M", f"{m_val}"]))

                acc = c_tree.get_accuracy(instance_validation.examples)
                sz = c_tree.get_nodes()

                print(f"m {m_val}, c {c_val}: {acc}, size {sz}")

                return acc, sz

            best_c = 0.25
            best_n = 3
            best_m = 2
            best_accuracy, _ = get_accuracy(best_c, best_m)

            c_c = 0.01
            max_m = len(instance.examples) // 5 * 4
            m_values = [1, 2, 3, 4, *[x for x in range(5, max_m + 1, 5)]]
            c_m = 1

            while c_c < 0.5:
                last_accuracies = deque(maxlen=5)
                for c_m in m_values:
                    accuracy, new_sz = get_accuracy(c_c, c_m)
                    if accuracy < 0.001:
                        break

                    if new_sz == 1 or (len(last_accuracies) >= 5 and all(x < best_accuracy for x in last_accuracies)):
                        break

                    last_accuracies.append(accuracy)

                    if accuracy >= best_accuracy:
                        best_c = c_c
                        best_m = c_m
                        best_accuracy = accuracy

                c_c += 0.01 if c_c < 0.05 else 0.05

            try:
                instance, instance_test, instance_validation = parse(pth, fl, c_slice, use_validation=True)
            except FileNotFoundError:
                # Invalid slice for instances with test set.
                continue

            instance.export_c45("/tmp/weka_instance.data")
            tree = parse_weka_tree(get_tree("/tmp/weka_instance.data", ["-B", "-C", f"{best_c}", "-M", f"{best_m}"]))
            with open(output_path, "w") as outp:
                outp.write(tree.as_string())
            print(f"Final accuracy {tree.get_accuracy(instance_test.examples)}")
