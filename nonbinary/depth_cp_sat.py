import itertools
from collections import defaultdict
from sys import maxsize, stdout
from threading import Timer

from pysat.formula import IDPool
from pysat.card import ITotalizer
from decision_tree import DecisionTree
import time


def _init_var(instance, limit, class_map):
    pool = IDPool()
    s = {}
    for xl in range(0, len(instance.examples)):
        s[xl] = {}
        for x2 in range(1, 2**limit):
            s[xl][x2] = pool.id(f"s{xl}_{x2}")

    t = defaultdict(list)

    f = {}
    for i in range(1, 2**limit):
        f[i] = {}
        for cf in range(1, instance.num_features + 1):
            if len(instance.domains[cf]) == 0:
                continue

            f[i][cf] = pool.id(f"f{i}_{cf}")

            if cf in instance.is_categorical:
                for _ in range(0, len(instance.domains[cf]) - len(t[i])):
                    t[i].append(pool.id(f"t{i}_{len(t[i])}"))

    c = {}
    for i in range(0, 2**limit):
        c[i] = {}
        if len(class_map) <= 2:
            c[i] = pool.id(f"c{i}")
        else:
            for j in range(1, len(class_map)+1):
                c[i][j] = pool.id(f"c{i}_{j}")

    z = {}
    for xl in range(0, len(instance.examples)):
        z[xl] = {}
        for i in range(0, 2**limit):
            z[xl][i] = pool.id(f"z{xl}_{i}")

    return s, f, c, z, t, pool


def encode(instance, limit, solver, opt_size=False):
    classes = list(instance.classes)  # Give classes an order

    if opt_size:
        classes.insert(0, "EmptyLeaf")

    if len(classes) == 2 or opt_size:
        class_map = {cc: i for i, cc in enumerate(classes)}
    else:
        class_map = {cc: i + 1 for i, cc in enumerate(classes)}

    s, f, c, z, t, p = _init_var(instance, limit, class_map)

    # each node has a feature
    for i in range(1, 2**limit):
        clause = []
        for f1, f1v in f[i].items():
            clause.append(f1v)
            for f2, f2v in f[i].items():
                if f2 > f1:
                    solver.add_clause([-f1v, -f2v])
        solver.add_clause(clause)

    # each leaf has a class
    if len(class_map) > 2:
        for c_n in range(0, 2 ** limit):
            clause = []

            for c1, c1v in c[c_n].items():
                clause.append(c1v)
                for c2, c2v in c[c_n].items():
                    if c2 > c1:
                        solver.add_clause([-c1v, -c2v])

            if not opt_size:
                solver.add_clause(clause)

    idx = list(range(0, len(instance.examples)))

    for cf in range(1, instance.num_features + 1):
        if cf not in instance.is_categorical:
            idx.sort(key=lambda x: (instance.examples[x].features[cf], x))

        for n in range(1, 2**limit):
            for i in range(0, len(t[n])):
                for j in range(i+1, len(t[n])):
                    solver.add_clause([-t[n][i], -t[n][j]])

            if cf in f[n]:
                if cf in instance.is_categorical:
                    for i in range(0, len(instance.examples)):
                        f_val = instance.examples[i].features[cf] if instance.examples[i].features[cf] != "?" else instance.domains_max[cf]
                        v_idx = instance.domains[cf].index(f_val)
                        solver.add_clause([-f[n][cf], -s[i][n], t[n][v_idx]])
                        solver.add_clause([-f[n][cf], s[i][n], -t[n][v_idx]])
                else:
                    solver.add_clause([-f[n][cf], s[idx[0]][n]])
                    if instance.examples[idx[0]].features[cf] != instance.examples[idx[-1]].features[cf]:
                        solver.add_clause([-f[n][cf], -s[idx[-1]][n]])

                    for i in range(0, len(instance.examples)-1):
                        idx1 = idx[i]
                        idx2 = idx[i+1]

                        f_val1 = instance.examples[idx1].features[cf] if instance.examples[idx1].features[cf] != "?" else \
                        instance.domains_max[cf]
                        f_val2 = instance.examples[idx2].features[cf] if instance.examples[idx2].features[
                                                                             cf] != "?" else \
                            instance.domains_max[cf]
                        if f_val1 == f_val2:
                            solver.add_clause([-f[n][cf], -s[idx1][n], s[idx2][n]])
                        solver.add_clause([-f[n][cf], s[idx1][n], -s[idx2][n]])

    def find_leaf(left_nodes, right_nodes, d, nid):
        if d == limit:
            cl = nid - 2**limit
            for i in range(0, len(instance.examples)):
                for nn in left_nodes:
                    solver.add_clause([-z[i][cl], s[i][nn]])
                for nn in right_nodes:
                    solver.add_clause([-z[i][cl], -s[i][nn]])
                cls = [z[i][cl]]
                cls.extend([-s[i][x] for x in left_nodes])
                cls.extend([s[i][x] for x in right_nodes])
                solver.add_clause(cls)

                c_vars = class_map[instance.examples[i].cls]
                if len(class_map) <= 2:
                    if c_vars == 0:
                        solver.add_clause([-z[i][cl], -c[cl]])
                    else:
                        solver.add_clause([-z[i][cl], c[cl]])
                else:
                    solver.add_clause([-z[i][cl], c[cl][c_vars]])
        else:
            left_nodes.append(nid)
            find_leaf(left_nodes, right_nodes, d+1, 2 * nid + 1)
            left_nodes.pop()
            right_nodes.append(nid)
            find_leaf(left_nodes, right_nodes, d+1, 2 * nid)
            right_nodes.pop()

    find_leaf([], [], 0, 1)

    return {"f": f, "s": s, "c": c, "z": z, "class_map": class_map, "pool": p}


def encode_size(vs, instance, solver, dl):
    pool = vs["pool"]
    card_vars = []
    c = vs["c"]

    for c_n in range(0, 2 ** dl):
        card_vars.append(pool.id(f"n{c_n}"))
        for c_c in c[c_n].values():
            solver.add_clause([-c_c, pool.id(f"n{c_n}")])

    return card_vars


def encode_extended_leaf_size(vs, instance, solver, dl):
    pool = vs["pool"]
    card_vars = []
    c = vs["c"]
    cm = vs["class_map"]

    for c_n in range(0, 2 ** dl):
        for cls, vals in cm.items():
            if not cls.startswith("-") and cls != "EmptyLeaf":
                solver.add_clause([-c[c_n][vals], pool.id(f"e{c_n}")])

        card_vars.append(pool.id(f"e{c_n}"))
    return card_vars


def encode_extended_leaf_limit(vs, solver, dl):
    c = vs["c"]
    cm = vs["class_map"]

    for c_n in range(0, 2 ** dl):
        for c_n2 in range(c_n + 1, 2 ** dl):
            for cls, vals in cm.items():
                if not cls.startswith("-") and cls != "EmptyLeaf":
                        solver.add_clause([-c[c_n][vals], -c[c_n2][vals]])


def estimate_size_add(instance, dl):
    c = len(instance.classes)
    return 2 ** dl * c * 2 + (2 ** dl) ** 2 * 3


def _decode(model, instance, limit, vs):
    class_map = vs['class_map']
    fs = vs["f"]
    cs = vs["c"]
    ss = vs["s"]
    zs = vs["z"]

    num_leafs = 2**limit
    tree = DecisionTree()
    assigned_fs = {}

    idx = list(range(0, len(instance.examples)))
    is_sorted = False

    paths = [set() for _ in range(0, len(instance.examples))]
    for i in range(0, len(instance.examples)):
        for n in range(0, 2**limit):
            if model[zs[i][n]]:
                c_id = 2**limit + n
                while c_id > 0:
                    assert c_id == 1 or (model[ss[i][c_id // 2]] == (False if c_id % 2 == 0 else True))
                    paths[i].add(c_id)
                    c_id //= 2
                break

    # Find features
    for f in range(1, instance.num_features + 1):
        is_sorted = False
        for i in range(1, num_leafs):
            if model[fs[i][f]]:
                if i in assigned_fs:
                    raise RuntimeError(f"ERROR: double features found for node {i}, features {f} and {tree.nodes[i].feature}")

                assigned_fs[i] = [f, None]
                if not is_sorted and f not in instance.is_categorical:
                    idx.sort(key=lambda x: (instance.examples[x].features[f], x))
                    is_sorted = True

                for c_idx, c_e in enumerate(idx):
                    if f in instance.is_categorical:
                        if model[ss[c_e][i]]:
                            assigned_fs[i] = (f, instance.examples[c_e].features[f])
                            break
                    else:
                        if not model[ss[c_e][i]]:
                            assert c_idx != 0
                            prev_idx = c_idx-1
                            assert instance.examples[idx[c_idx-1]].features[f] < instance.examples[c_e].features[f]
                            # while instance.examples[idx[prev_idx]].features[f] == instance.examples[c_e].features[f]:
                            #     prev_idx = prev_idx - 1

                            assigned_fs[i] = (f, instance.examples[idx[prev_idx]].features[f])
                            break

    for i in range(1, num_leafs):
        a_f, a_v = assigned_fs[i]
        if a_v is None:
            a_v = next(iter(instance.domains[a_f]))
            print(f"No path through node {i}")

        if i == 1:
            tree.set_root(a_f, a_v, a_f in instance.is_categorical)
        else:
            tree.add_node(a_f, a_v, i // 2, i % 2 == 1, a_f in instance.is_categorical)

    rev_lookup = {v: k for k, v in class_map.items()}
    for i in range(0, num_leafs):
        c_c = None
        if len(class_map) <= 2:
            if not bool(model[cs[i]]):
                c_c = rev_lookup[0]
            elif len(class_map) > 1:
                c_c = rev_lookup[1]
        else:
            for c_i in range(1, len(class_map)+1):
                if bool(model[cs[i][c_i]]):
                    c_c = rev_lookup[c_i]
                    break

        if num_leafs == 1:
            tree.set_root_leaf(c_c)
        else:
            tree.add_leaf(c_c, (num_leafs + i) // 2, i % 2 == 1)

    tree.get_accuracy(instance.examples)
    tree.clean(instance)

    return tree


def new_bound(tree, instance):
    if tree is None:
        return 1

    def dfs_find(node, level):
        if node.is_leaf:
            return level
        else:
            return max(dfs_find(node.left, level + 1), dfs_find(node.right, level + 1))

    return dfs_find(tree.root, 0)


def lb():
    return 1


def increment():
    return 1


def is_sat():
    return True


def estimate_size(instance, depth):
    """Estimates the required size in the number of literals"""

    d2 = 2**depth
    c = len(instance.classes)
    s = len(instance.examples)
    f = instance.num_features
    md = 0
    if len(instance.is_categorical) > 0:
        md = max(len(instance.domains[cf]) for cf in instance.is_categorical)

    return d2 * f * (f-1) // 2 + 2*c*c*d2 + f * d2 * md * 2 + f * d2 * s*6


def get_tree_size(tree):
    return tree.root.get_leaves()
