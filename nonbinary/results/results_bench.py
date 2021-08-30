import sys
import tarfile
from os import linesep
from sklearn.linear_model import LinearRegression
from sys import maxsize
from decimal import Decimal, InvalidOperation
from nonbinary.decision_tree import DecisionTree, DecisionTreeLeaf, DecisionTreeNode
import os
import re

import subprocess

weka_path = os.path.join(os.path.expanduser("~"), "Downloads/weka-3-8-5-azul-zulu-linux/weka-3-8-5")
jre_path = os.path.join(weka_path, "jre/zulu11.43.55-ca-fx-jre11.0.9.1-linux_x64/bin/java")
main_pth = os.path.abspath("./")

fields = [
    "reduced",
    "encoding_size",
    "depth",
    "sum_domain",
    "max_domain",
    "examples",
    "classes",
    "features",
    "entropy"
]

field_map = {x: i+1 for i, x in enumerate(fields)}


class BenchmarkResult:
    def __init__(self, instance, instance_slice, duration, timeout, depth, examples, classes, features, max_domain, sum_domain, encoding_size, reduced, entropy):
        self.instance = instance
        self.slice = instance_slice
        self.duration = duration
        self.timeout = timeout
        self.depth = depth
        self.examples = examples
        self.classes = classes
        self.features = features
        self.max_domain = max_domain
        self.sum_domain = sum_domain
        self.encoding_size = encoding_size
        self.reduced = reduced
        self.entropy = entropy


class WekaNode:
    def __init__(self, feat=None, threshold=None, is_cat=None, cls=None):
        self.cls = cls
        self.feat = feat
        self.threshold = [threshold]
        self.children = []
        self.is_cat = is_cat



def parse_file(c_file):
    instance_name = None
    instance_slice = None

    results = []
    for i, cl in enumerate(c_file):
        if type(cl) is not str:
            cl = cl.decode('ascii')
        if i == 0:
            instance_name = cl.split(",")[0].split(":")[1].strip()
            slice_idx = cl.index("slice=")
            end_idx = cl[slice_idx:].index(",")
            instance_slice = cl[slice_idx+6: slice_idx+end_idx]
        if cl.startswith("Running"):
            c_bound = int(cl.split(",")[0].split(" ")[1])
        elif cl.startswith("E:"):
            fields = cl.replace(" True", "1").replace(" False", "0").replace("T:MO", f"T:{maxsize}").replace("T:TO", f"T:{maxsize}")\
                .split(" ")
            fields = [int(float(x.split(":")[1])) for x in fields]
            results.append(BenchmarkResult(instance_name, instance_slice, fields[1], False, fields[6], fields[0],
                                           fields[2], fields[3], fields[5], fields[4], fields[7],
                                            1 if fields[8] == "True" else 0, fields[9]))
    return results



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

                t_str = c_line.split(" ")[2].replace(":", "")  # remove the : at the end
                try:
                    threshold = int(t_str)
                except ValueError:
                    try:
                        threshold = Decimal(t_str)
                    except InvalidOperation:
                        threshold = t_str

                if depth > l_depth:
                    feature = field_map[c_line[0:c_line.find(" ")]]

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


with tarfile.open(sys.argv[1]) as tar_file:
    results = []
    for ctf in tar_file:
        file_parts = ctf.path.split(".")

        if file_parts[-2].startswith("e"):
            # Error file
            continue
        cetf = tar_file.extractfile(ctf)

        results.extend(parse_file(cetf))
#
# with open("benchmark_results.csv", "w") as outp:
#     outp.write("Instance;Slice;Reduced;Duration;Encoding Size;Depth;Domain Sizes;Examples;Classes;Features;Nodes;Decisions"+linesep)
#     for cr in results:
#         outp.write(f"{cr.instance};{cr.slice};{cr.reduced};{cr.duration};{cr.encoding_size};{cr.depth};{cr.sum_domain};{cr.examples};{cr.classes};{cr.features};{2**cr.depth};{2**cr.depth * cr.sum_domain}"+linesep)

with open("benchmark_results.names", "w") as outp:
    outp.write("0, 1."+linesep)
    outp.write("reduced: 0,1."+ linesep)
    outp.write("encoding_size: continuous."+ linesep)
    outp.write("depth: continuous."+ linesep)
    outp.write("sum_domain: continuous."+ linesep)
    outp.write("max_domain: continuous."+ linesep)
    outp.write("examples: continuous."+ linesep)
    outp.write("classes: continuous."+ linesep)
    outp.write("features: continuous."+ linesep)
    outp.write("entropy: continuous."+ linesep)

with open("benchmark_results.data", "w") as outp:
    for cr in results:
        outp.write(f"{cr.reduced},{cr.encoding_size},{cr.depth},{cr.sum_domain},{cr.max_domain},{cr.examples},{cr.classes},{cr.features},{cr.entropy},{0 if cr.duration > 330 else 1}."+linesep)

# X = []
# y = []
#
# for cr in results:
#     if cr.reduced == 1:
#         X.append([cr.depth, 2**cr.depth, cr.sum_domain, cr.classes, cr.encoding_size, cr.reduced, cr.examples, 2**cr.depth * cr.sum_domain])
#         # X.append([2 ** cr.depth, cr.sum_domain, cr.reduced, cr.examples,
#         #           2 ** cr.depth * cr.sum_domain, cr.encoding_size])
#         y.append(cr.duration)
#
# reg = LinearRegression().fit(X, y)
# #reg = Lasso(alpha=0.1).fit(X, y)
# print(f"{reg.score(X,y)}")
# print(f"{reg.coef_}")


process = subprocess.Popen([
        jre_path, "-cp", os.path.join(weka_path, "weka.jar"),
        "weka.classifiers.trees.J48",
        "-t", os.path.join(main_pth, "benchmark_results.data"), "-no-cv",
        "-C", "0.25", "-M", "2"],
        cwd=weka_path,
        stderr=open(os.devnull, 'w'),
        stdout=subprocess.PIPE
        )

output, _ = process.communicate()
output = output.decode('ascii')

mt = re.search("J48 u?n?pruned tree[^\-]*[\-]*(.*)Number of Leaves", output, re.DOTALL)


new_tree = parse_weka_tree(mt.group(1).strip().splitlines())
with open("../benchmark_tree.dt", "w") as outp:
    outp.write(new_tree.as_string())
