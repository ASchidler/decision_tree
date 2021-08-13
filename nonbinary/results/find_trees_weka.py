import os
import shutil
import subprocess
import re
from sys import maxsize
import improve.tree_parsers as tp
import parser
from nonbinary.nonbinary_instance import ClassificationInstance, Example, parse
from collections import deque
from nonbinary.decision_tree import DecisionTree, DecisionTreeLeaf, DecisionTreeNode
from decimal import Decimal, InvalidOperation

use_validation = False
pruning = 1  # 0 is no pruning

pth = "nonbinary/instances"
weka_path = os.path.join(os.path.expanduser("~"), "Downloads/weka-3-8-5-azul-zulu-linux/weka-3-8-5")
jre_path = os.path.join(weka_path, "jre/zulu11.43.55-ca-fx-jre11.0.9.1-linux_x64/bin/java")
#-cp ./weka.jar

# Only for non pruned trees
parameters = ["-U", "-J", "-M", "0"]


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
        stderr=open(os.devnull, 'w'),
        stdout=subprocess.PIPE
        )

    output, _ = process.communicate()
    output = output.decode('ascii')

    mt = re.search("J48 u?n?pruned tree[^\-]*[\-]*(.*)Number of Leaves", output, re.DOTALL)


    return mt.group(1).strip().splitlines()


fls = {".".join(x.split(".")[:-2]) for x in list(os.listdir(pth)) if x.endswith(".data")}
fls = sorted(fls)

for fl in fls:
    print(f"{fl}")
    for c_slice in range(1, 6):
        print(f"{c_slice}")

        if pruning == 0:
            fld = "unpruned" if not use_validation else "validation"
            output_path = f"nonbinary/results/trees/{fld}/{fl}.{c_slice}.w.dt"
        else:
            output_path = f"nonbinary/results/trees/pruned/{fl}.{c_slice}.w.dt"
        if os.path.exists(output_path):
            continue

        try:
            instance, instance_test, instance_validation = parse(pth, fl, c_slice, use_validation=use_validation or pruning == 1)
        except FileNotFoundError:
            # Invalid slice for instances with test set.
            continue

        instance.export_c45("/tmp/weka_instance.data")

        if pruning == 0:
            tree = parse_weka_tree(get_tree("/tmp/weka_instance.data", parameters))
            with open(output_path, "w") as outp:
                outp.write(tree.as_string())
        else:
            def get_accuracy(c_val, m_val):
                c_tree = parse_weka_tree(get_tree("/tmp/weka_instance.data", ["-C", f"{c_val}", "-M", f"{m_val}"]))

                acc = c_tree.get_accuracy(instance_validation.examples)
                sz = c_tree.get_nodes()

                print(f"m {m_val}, c {c_val}: {acc}, size {sz}")

                return acc, sz

            best_c = 0.25
            best_n = 3
            best_m = 2
            best_accuracy, _ = get_accuracy(best_c, best_m)

            c_c = 0.01
            while c_c < 0.5:
                c_accuracy, _ = get_accuracy(c_c, best_m)

                if c_accuracy > best_accuracy:
                    best_accuracy = c_accuracy
                    best_c = c_c

                if c_c < 0.05:
                    c_c += 0.01
                else:
                    c_c += 0.05

            max_m = len(instance.examples) // 5 * 4
            m_values = [1, 2, 3, 4, *[x for x in range(5, max_m+1, 5)]]
            c_m = 1
            last_accuracies = deque(maxlen=5)

            for c_m in m_values:
                c_accuracy, c_sz = get_accuracy(best_c, c_m) if pruning == 1 else get_accuracy(best_n, c_m)

                if c_accuracy > best_accuracy:
                    best_accuracy = c_accuracy
                    best_m = c_m
                elif (c_sz == 1) or (
                        len(last_accuracies) >= 5 and all(x < best_accuracy for x in last_accuracies)):
                    break

                last_accuracies.append(c_accuracy)

            try:
                instance, instance_test, instance_validation = parse(pth, fl, c_slice, use_validation=False)
            except FileNotFoundError:
                # Invalid slice for instances with test set.
                continue

            instance.export_c45("/tmp/weka_instance.data")
            tree = parse_weka_tree(get_tree("/tmp/weka_instance.data", ["-C", f"{best_c}", "-M", f"{best_m}"]))
            with open(output_path, "w") as outp:
                outp.write(tree.as_string())
            print(f"Final accuracy {tree.get_accuracy(instance_test.examples)}")
