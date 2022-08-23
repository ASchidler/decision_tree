import os
import subprocess
import re
from sklearn import tree
from nonbinary.decision_tree import DecisionTree, DecisionTreeLeaf, DecisionTreeNode
from collections import defaultdict
from nonbinary.nonbinary_instance import ClassificationInstance, Example, parse
from numpy import nan, isnan
from sklearn.impute import SimpleImputer
import nonbinary.tree_parsers as tp
from decimal import Decimal


use_validation = False
pruning = 1  # 0 is no pruning

pth = "nonbinary/instances"
#fls = {".".join(x.split(".")[:-2]) for x in list(os.listdir(pth)) if x.endswith("primary-tumor.1.data")}
#fls = {".".join(x.split(".")[:-2]) for x in list(os.listdir(pth)) if x.endswith("audiology.1.data")}
fls = {".".join(x.split(".")[:-2]) for x in list(os.listdir(pth)) if x.endswith(".data")}
fls = sorted(fls)


def convert_instance(c_domains, c_inst, feat_map=None, v_map=None, cat_map=None):
    c_y = [k.cls for k in c_inst.examples]

    # First step add missing values
    t_x = []
    domains = []
    missing = set()

    if not feat_map:
        feat_map = dict()
        for c_f in range(1, c_inst.num_features+1):
            if len(c_inst.domains[c_f]) > 0:
                feat_map[c_f] = len(feat_map)
                domains.append(set(c_domains[c_f]))
    else:
        for k in feat_map.keys():
            domains.append(set(c_domains[k]))

    for c_e in c_inst.examples:
        c_arr = []
        for c_f in range(1, c_inst.num_features+1):
            if c_f in feat_map:
                if c_e.features[c_f] == "?":
                    missing.add(c_f)
                c_arr.append(c_e.features[c_f])
        t_x.append(c_arr)

    if c_inst.has_missing:
        for c_f in missing:
            if len(c_inst.domains[c_f]) > 0:
                c_arr = []
                for c_e in c_inst.examples:
                    c_arr.append([c_e.features[c_f] if c_e.features[c_f] != "?" else nan])

                imp = SimpleImputer(missing_values=nan, strategy=("mean" if c_f not in c_inst.is_categorical else "most_frequent"))
                imp.fit(c_arr, c_y)
                c_arr = imp.transform(c_arr)

                for ci, entry in enumerate(c_arr):
                    t_x[ci][feat_map[c_f]] = entry[0]
            else:
                if c_f in c_inst.is_categorical:
                    for c_arr in t_x:
                        c_arr[feat_map[c_f]] = nan
                else:
                    for c_arr in t_x:
                        c_arr[feat_map[c_f]] = 0


    # Second step map categorical values
    r_map = None
    r_feat_map = {v: k for k, v in feat_map.items()}
    if v_map is None:
        r_map = {}
        cat_map = {}
        v_map = {}

        c_idx = 0
        for ci in range(0, len(t_x[0])):
            if r_feat_map[ci] in c_inst.is_categorical:
                cat_map[ci] = c_idx
                v_map[ci] = dict()
                for c_v in domains[ci]:
                    v_map[ci][c_v] = c_idx
                    r_map[c_idx] = (ci, c_v)
                    c_idx += 1
            else:
                cat_map[ci] = c_idx
                r_map[c_idx] = ci
                c_idx += 1

    c_x = []
    for c_arr in t_x:
        new_arr = []
        for ci in range(0, len(c_arr)):
            if r_feat_map[ci] not in c_inst.is_categorical:
                new_arr.append(c_arr[ci])
            else:
                new_arr.extend([0 for _ in range(0, len(domains[ci]))])
                if c_arr[ci] in v_map[ci]:
                    new_arr[v_map[ci][c_arr[ci]]] = 1
        c_x.append(new_arr)

    return c_x, c_y, feat_map, cat_map, v_map, r_map


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

        all_domains = [set() for x in range(0, instance.num_features + 1)]
        for i in range(1, instance.num_features + 1):
            all_domains[i].update(instance.domains[i])
            instance.is_categorical.update(instance_test.is_categorical)
            instance_test.is_categorical.update(instance.is_categorical)
            all_domains[i].update(instance_test.domains[i])
            if instance_validation:
                all_domains[i].update(instance_validation.domains[i])
                instance.is_categorical.update(instance_validation.is_categorical)
                instance_validation.is_categorical.update(instance.is_categorical)

        x, y, f_map, c_map, v_map, r_map = convert_instance(all_domains, instance)

        if pruning == 0:
            cls = tree.DecisionTreeClassifier()
            cls.fit(x, y)
        else:
            full_instance, _, _ = parse(pth, fl, c_slice, use_validation=True)
            full_x, full_y, _, _, _, _ = convert_instance(all_domains, full_instance, f_map, v_map, c_map)
            val_x, val_y, _, _, _, _ = convert_instance(all_domains, instance_validation, f_map, v_map, c_map)

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
        test_x, test_y, _, _, _, _ = convert_instance(all_domains, instance_test, f_map, v_map, c_map)
        print(f"Computed {fl}, accuracy: {cls.score(x, y)}/{cls.score(test_x, test_y)}")
        new_tree = DecisionTree()
        classes = list(cls.classes_)
        nodes = []

        r_f_map = {v: k for k, v in f_map.items()}
        for i in range(0, len(cls.tree_.feature)):
            if cls.tree_.feature[i] >= 0:
                nf = cls.tree_.feature[i]
                map_val = r_map[nf]

                if isinstance(map_val, tuple):
                    nf = r_f_map[map_val[0]]
                    ts = map_val[1]
                    is_cat = True
                else:
                    nf = r_f_map[map_val]
                    ts = Decimal(int(cls.tree_.threshold[i] * 1000000)) / Decimal(1000000.0)
                    is_cat = False

                nodes.append(DecisionTreeNode(nf, ts, i, None, is_cat))
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

        # CART treats categorical the other way round, rectify...
        def switch_nodes(c_n):
            if not c_n.is_leaf:
                if c_n.is_categorical:
                    c_n.left, c_n.right = c_n.right, c_n.left
                switch_nodes(c_n.left)
                switch_nodes(c_n.right)

        construct_tree(nodes[0], None, None)
        switch_nodes(new_tree.root)
        print(f"Accuracy: {new_tree.get_accuracy(instance.examples)}/{new_tree.get_accuracy(instance_test.examples)} {new_tree.get_depth()} {new_tree.get_nodes()}")
        #
        # def find_features(c_n):
        #     if not c_n.is_leaf:
        #         fst, tsh = find_features(c_n.left)
        #         fst2, tsh2 = find_features(c_n.right)
        #         fst.update(fst2)
        #         tsh.update(tsh2)
        #         fst.add(c_n.feature)
        #         tsh.add(c_n.threshold)
        #         return fst, tsh
        #     else:
        #         return set(), set()
        #
        # features, thresholds = find_features(new_tree.root)
        # new_instance = ClassificationInstance()
        # for i, c_arr in enumerate(x):
        #     c_e = [None for _ in range(0, instance.num_features + 1)]
        #     for cidx in range(0, len(c_arr)):
        #         mapv = r_map[cidx]
        #         if isinstance(mapv, tuple):
        #             if c_arr[cidx] == 1:
        #                 c_e[r_f_map[mapv[0]]] = mapv[1]
        #         else:
        #             c_e[r_f_map[mapv]] = c_arr[cidx]
        #
        #     for cidx in features:
        #         c_val = instance.examples[i].features[cidx]
        #         if c_val == "?" and len(instance.domains[cidx]) != 0:
        #             c_val = instance.domains_max[cidx]
        #
        #         if c_e[cidx] is None and cidx in features:
        #             print("Needed value")
        #
        #         if c_val != c_e[cidx] and c_e[cidx] is not None:
        #             print(f"mismatch {instance.examples[i].features[cidx]} {c_e[cidx]}")
        #         elif c_e[cidx] is not None:
        #             instance.examples[i].features[cidx] = c_val
        #         if instance.examples[i].cls != y[i]:
        #             print("Class mismatch")
        #
        #     new_instance.add_example(Example(new_instance, c_e[1:], y[i]))
        #
        #
        # print(f"{new_tree.get_accuracy(new_instance.examples)}")
        # with open("/tmp/tmp.dt", "w") as outp:
        #     outp.write(new_tree.as_string())
        # new_tree = tp.parse_internal_tree("/tmp/tmp.dt")
        # print(f"{new_tree.get_accuracy(instance.examples)}")
        # print("Done")

        with open(output_path, "w") as outf:
            outf.write(new_tree.as_string())
