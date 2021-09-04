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
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
from nonbinary.nonbinary_instance import ClassificationInstance, Example

weka_path = os.path.join(os.path.expanduser("~"), "Downloads/weka-3-8-5-azul-zulu-linux/weka-3-8-5")
jre_path = os.path.join(weka_path, "jre/zulu11.43.55-ca-fx-jre11.0.9.1-linux_x64/bin/java")
main_pth = os.path.abspath("./")

create_encoding_benchmark = True

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

maximum_timelimit = 300

field_map = {x: i+1 for i, x in enumerate(fields)}


class BenchmarkResult:
    def __init__(self, instance, instance_slice, duration, timeout, depth, examples, classes, features, max_domain, sum_domain, encoding_size, reduced, entropy, unsat):
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
        self.unsat = unsat


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

    enc_results = [[] for _ in range(0, 4)]
    c_encoding = 1
    results = []
    c_enc_results = [[] for _ in range(0, 4)]

    for i, cl in enumerate(c_file):
        if type(cl) is not str:
            cl = cl.decode('ascii')
        if i == 0:
            instance_name = cl.split(",")[0].split(":")[1].strip()
            slice_idx = cl.index("slice=")
            end_idx = cl[slice_idx:].index(",")
            instance_slice = cl[slice_idx+6: slice_idx+end_idx]
        if cl.startswith("Testing Encoding"):
            c_encoding = int(cl.strip()[-1])
        elif cl.startswith("E:"):
            fields = cl.replace(" True", "1").replace(" False", "0") \
                .replace("True", "1").replace("False", "0") \
                .replace("T:MO", f"T:{maxsize}").replace("T:TO", f"T:{maxsize}")\
                .split(" ")
            is_opt = False
            if fields[1].find("*") > -1:
                is_opt = True
                fields[1] = fields[1].replace("*", "")
            fields = [int(float(x.split(":")[1])) for x in fields]
            enc_results[c_encoding].append(BenchmarkResult(instance_name, instance_slice, fields[1], False, fields[6], fields[0],
                                           fields[2], fields[3], fields[5], fields[4], fields[7],
                                            fields[8], fields[9], is_opt))
            c_enc_results[c_encoding].append(enc_results[c_encoding][-1])
        elif cl.startswith("Time"):
            c_enc_results2 = [[]]
            for encoding_idx in range(1, len(c_enc_results)):
                c_enc_results2.append([x for x in c_enc_results[encoding_idx] if x.duration <= maximum_timelimit])

            best_encoding = 1
            best_result = None
            for encoding_idx in range(1, 4):
                if len(c_enc_results2[encoding_idx]) > 0:
                    c_result_entry = c_enc_results2[encoding_idx][-1]
                    is_better = best_result is None or \
                        (c_result_entry.unsat and not best_result.unsat) \
                        or c_result_entry.depth < best_result.depth \
                        or (c_result_entry.depth == best_result.depth and c_result_entry.duration < best_result.duration)

                    if is_better:
                        best_encoding = encoding_idx
                        best_result = c_result_entry

            # Example for trying to solve the instance
            if best_result is not None:
                results.append((best_result, best_encoding))
            # Example for not trying to solve instance
            else:
                c_result = None
                for encoding_idx in range(1, 4):
                    for c_entry in c_enc_results[encoding_idx]:
                        if c_result is None or c_result.depth > c_entry.depth:
                            c_result = c_entry
                if c_result is not None:
                    results.append((c_result, -1))

            c_enc_results = [[] for _ in range(0, 4)]
    return results, enc_results



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


final_enc_results = []
with tarfile.open(sys.argv[1]) as tar_file:
    results = []
    for ctf in tar_file:
        file_parts = ctf.path.split(".")

        if file_parts[-2].startswith("e"):
            # Error file
            continue
        cetf = tar_file.extractfile(ctf)

        new_results, new_enc_results = parse_file(cetf)
        results.extend(new_results)
        final_enc_results.append(new_enc_results)
#
# with open("benchmark_results.csv", "w") as outp:
#     outp.write("Instance;Slice;Reduced;Duration;Encoding Size;Depth;Domain Sizes;Examples;Classes;Features;Nodes;Decisions"+linesep)
#     for cr in results:
#         outp.write(f"{cr.instance};{cr.slice};{cr.reduced};{cr.duration};{cr.encoding_size};{cr.depth};{cr.sum_domain};{cr.examples};{cr.classes};{cr.features};{2**cr.depth};{2**cr.depth * cr.sum_domain}"+linesep)
#

with open("benchmark_results_encodings.names" if create_encoding_benchmark else "benchmark_results.names", "w") as outp:
    if create_encoding_benchmark:
        outp.write("-1, 1, 2, 3." + linesep)
    else:
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

X = []
y = []

if create_encoding_benchmark:
    with open("benchmark_results_encodings.data", "w") as outp:
        for cr, recommendation in results:
            y.append(recommendation)
            X.append([cr.reduced,cr.encoding_size,cr.depth,cr.sum_domain,cr.max_domain,cr.examples,cr.classes,cr.features,cr.entropy])
            outp.write(",".join([str(x) for x in X[-1]]) + "," + str(y[-1]) + linesep)
else:
    with open("benchmark_results.data", "w") as outp:
        for c_batch in final_enc_results:
            for cr in c_batch[1]:
                y.append(1 if cr.duration <= maximum_timelimit else 0)
                X.append([cr.reduced,cr.encoding_size,cr.depth,cr.sum_domain,cr.max_domain,cr.examples,cr.classes,cr.features,cr.entropy])
                outp.write(",".join([str(x) for x in X[-1]]) + "," + str(y[-1]) + linesep)


cls = tree.DecisionTreeClassifier()
cls.fit(X, y)
path = cls.cost_complexity_pruning_path(X, y)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

indices = StratifiedKFold(shuffle=True).split(X, y)
splits = []

for idx, idx_test in indices:
    c_X = [X[c_idx] for c_idx in idx]
    c_y = [y[c_idx] for c_idx in idx]
    c_Xt = [X[c_idx] for c_idx in idx_test]
    c_yt = [y[c_idx] for c_idx in idx_test]
    splits.append((c_X, c_y, c_Xt, c_yt))

alpha_precision = {}
for c_alpha in ccp_alphas:
    alpha_results = 0
    for c_X, c_y, c_Xt, c_yt in splits:
        cls_tmp = tree.DecisionTreeClassifier(ccp_alpha=c_alpha)
        cls_tmp.fit(c_X, c_y)
        alpha_results += cls_tmp.score(c_Xt, c_yt)

    alpha_precision[c_alpha] = alpha_results
    print(f"alpha {c_alpha}: {alpha_results / len(splits)}")
best_alpha, best_acc = max(alpha_precision.items(), key=lambda x: x[1])
print(f"Accuracy: {best_acc / 5}")


cls = tree.DecisionTreeClassifier(ccp_alpha=best_alpha)
cls.fit(X, y)

print(f"{cls.score(X, y)}")

new_tree = DecisionTree()
classes = list(cls.classes_)
nodes = []

for i in range(0, len(cls.tree_.feature)):
    if cls.tree_.feature[i] >= 0:
        nf = cls.tree_.feature[i]
        ts = Decimal(int(cls.tree_.threshold[i] * 1000000)) / Decimal(1000000.0)
        nodes.append(DecisionTreeNode(nf+1, ts, i, None, False))
    else:
        c_max = (-1, None)
        for cc in range(0, len(classes)):
            c_max = max(c_max, (cls.tree_.value[i][0][cc], cc))
        nodes.append(DecisionTreeLeaf(str(classes[c_max[1]]), i, None))

def construct_tree(c_n, parent, pol):
    if c_n.is_leaf:
        if parent is None:
            new_tree.set_root_leaf(c_n.cls)
        else:
            new_tree.add_leaf(c_n.cls, parent, pol)
    else:
        if parent is None:
            nn = new_tree.set_root(c_n.feature, c_n.threshold, c_n.is_categorical)
        else:
            nn = new_tree.add_node(c_n.feature, c_n.threshold, parent, pol, c_n.is_categorical)

        construct_tree(nodes[cls.tree_.children_left[c_n.id]], nn.id, True)
        construct_tree(nodes[cls.tree_.children_right[c_n.id]], nn.id, False)

# CART treats categorical the other way round, rectify...
def switch_nodes(c_n):
    if not c_n.is_leaf:
        if c_n.is_categorical:
            c_n.left, c_n.right = c_n.right, c_n.left
        switch_nodes(c_n.left)
        switch_nodes(c_n.right)

construct_tree(nodes[0], None, None)
switch_nodes(new_tree.root)

# Test accuracy of conversion
instance = ClassificationInstance()
for i, c_features in enumerate(X):
    instance.add_example(Example(instance, c_features, str(y[i])))
print(f"{new_tree.get_accuracy(instance.examples)}")

with open("../benchmark_tree_encodings.dt" if create_encoding_benchmark else "../benchmark_tree.dt", "w") as outp:
    outp.write(new_tree.as_string())
