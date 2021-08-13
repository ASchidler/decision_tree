import os
import subprocess
import re
from sklearn import tree
from nonbinary.decision_tree import DecisionTree, DecisionTreeLeaf, DecisionTreeNode
from collections import defaultdict
from nonbinary.nonbinary_instance import ClassificationInstance, Example, parse
from numpy import nan
from sklearn.impute import SimpleImputer

use_validation = False
pruning = 1 # 0 is no pruning

pth = "nonbinary/instances"
fls = {".".join(x.split(".")[:-2]) for x in list(os.listdir(pth)) if x.endswith(".data")}
fls = sorted(fls)


def convert_instance(c_inst):
    c_x = []
    maps = defaultdict(list)
    r_map = {}
    c_y = [k.cls for k in c_inst.examples]
    t_x = []

    domains = []
    missing = set()
    feat_map = dict()
    for c_f in range(1, c_inst.num_features+1):
        if len(c_inst.domains[c_f]) > 0:
            feat_map[c_f] = len(feat_map)
            domains.append(set(c_inst.domains[c_f]))

    for c_e in c_inst.examples:
        c_arr = []
        for c_f in range(1, c_inst.num_features+1):
            if len(c_inst.domains[c_f]) > 0:
                if c_e.features[c_f] == "?":
                    missing.add(c_f)
                c_arr.append(c_e.features[c_f])
        t_x.append(c_arr)

    if c_inst.has_missing:
        for c_f in missing:
            c_arr = []
            for c_e in c_inst.examples:
                c_arr.append([c_e.features[c_f] if c_e.features[c_f] != "?" else nan])

            imp = SimpleImputer(missing_values=nan, strategy=("mean" if c_f not in c_inst.is_categorical else "most_frequent"))
            imp.fit(c_arr, c_y)
            c_arr = imp.transform(c_arr)

            for ci, entry in enumerate(c_arr):
                domains[feat_map[c_f]].add(entry[0])
                t_x[ci][feat_map[c_f]] = entry[0]

    c_idx = 0


    c_idx = c_inst.num_features + 2
    for c_f in range(1, c_inst.num_features+1):
        if c_f in c_inst.is_categorical:
            for c_v in c_inst.domains[c_f]:
                maps[c_f].append((c_v, c_idx))
                r_map[c_idx] = (c_f, c_v)
                c_idx += 1

    for c_e in c_inst.examples:
        c_arr = []
        for c_f in range(1, c_inst.num_features+1):
            if c_f in c_inst.is_categorical:
                for c_v, c_idx in maps[c_f]:
                    if c_v == "?":
                        c_arr.append(nan)
                    elif c_v == c_e.features[c_f]:
                        c_arr.append(True)
                    else:
                        c_arr.append(False)
            else:
                if c_e.features[c_f] == "?":
                    c_arr.append(nan)
                    missing = True
                else:
                    c_arr.append(c_e.features[c_f])
        c_x.append(c_arr)

    c_y = [k.cls for k in c_inst.examples]
    if missing:
        imp = SimpleImputer()
        imp.fit(c_x)
        c_x = imp.transform(c_x)
    return c_x, c_y, r_map

for fl in fls:
    print(f"{fl}")
    for c_slice in range(1, 6):
        print(f"{c_slice}")

        if pruning == 0:
            fld = "unpruned" if not use_validation else "validation"
            output_path = f"nonbinary/results/trees/{fld}/{fl}.{c_slice}.c.dt"
        else:
            output_path = f"nonbinary/results/trees/pruned/{fl}.{c_slice}.c.dt"
        if os.path.exists(output_path):
            continue

        try:
            instance, instance_test, instance_validation = parse(pth, fl, c_slice, use_validation=use_validation or pruning == 1)
        except FileNotFoundError:
            # Invalid slice for instances with test set.
            continue

        x, y, rmap = convert_instance(instance)
        for c_d in enumerate(instance.domains):
            if len(c_d[1]) > 0:
                instance_test.domains[c_d[0]].extend(set(c_d[1]) - set(instance_test.domains[c_d[0]]))
        instance_test.is_categorical.update(instance.is_categorical)

        if pruning == 0:
            cls = tree.DecisionTreeClassifier()
            cls.fit(x, y)
        else:
            full_instance, _, _ = parse(pth, fl, c_slice, use_validation=False)
            full_x, full_y, rmap = convert_instance(full_instance)
            val_x, val_y, _ = convert_instance(instance_validation)

            for c_d in enumerate(full_instance.domains):
                if len(c_d[1]) > 0:
                    instance_test.domains[c_d[0]].extend(set(c_d[1]) - set(instance_test.domains[c_d[0]]))
                    instance_validation.domains[c_d[0]].extend(set(c_d[1]) - set(instance_validation.domains[c_d[0]]))
            instance_test.is_categorical.update(instance.is_categorical)
            instance_validation.is_categorical.update(instance.is_categorical)

            full_cls = tree.DecisionTreeClassifier()
            full_cls.fit(full_x, full_y)
            path = full_cls.cost_complexity_pruning_path(full_x, full_y)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities

            alpha_precision = dict()
            for c_alpha in ccp_alphas:
                cls_tmp = tree.DecisionTreeClassifier(ccp_alpha=c_alpha)
                cls_tmp.fit(x, y)
                alpha_precision[c_alpha] = cls_tmp.score(val_x, val_y)
                print(f"alpha {c_alpha}: {alpha_precision[c_alpha]}")
            best_alpha, _ = max(alpha_precision.items(), key=lambda x: x[1])

            cls = tree.DecisionTreeClassifier(ccp_alpha=best_alpha)
            cls.fit(full_x, full_y)

        # Convert
        test_x, test_y, _ = convert_instance(instance_test)
        print(f"Computed {fl}, accuracy: {cls.score(x, y)}/{cls.score(test_x, test_y)}")
        new_tree = DecisionTree()
        classes = list(cls.classes_)
        nodes = []

        for i in range(0, len(cls.tree_.feature)):
            if cls.tree_.feature[i] >= 0:
                nf = cls.tree_.feature[i]+1
                is_cat = nf in rmap
                nf = nf if not is_cat else rmap[nf]
                nodes.append(DecisionTreeNode(nf, cls.tree_.threshold[i], i, None, is_cat))
            else:
                c_max = (-1, None)
                for cc in range(0, len(classes)):
                    c_max = max(c_max, (cls.tree_.value[i][0][cc], cc))
                nodes.append(DecisionTreeLeaf(classes[c_max[1]].strip(), i, None))

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

        construct_tree(nodes[0], None, None)
        print(f"Accuracy: {new_tree.get_accuracy(instance.examples)}")

        # with open(output_path, "w") as outf:
        #     outf.write(new_tree.as_string())
