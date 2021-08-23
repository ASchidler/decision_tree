import itertools
from sys import maxsize, stdout
from threading import Timer

from pysat.formula import IDPool
from pysat.card import ITotalizer
from decision_tree import DecisionTree
import time


def _init_var(instance, limit, class_map):
    pool = IDPool()
    x = {}
    for xl in range(0, len(instance.examples)):
        x[xl] = {}
        for x2 in range(0, limit):
            x[xl][x2] = pool.id(f"x{xl}_{x2}")

    max_domain = 0
    d = {}
    for cf in range(1, instance.num_features+1):
        max_domain = max(max_domain, len(instance.domains[cf]) - (0 if cf in instance.is_categorical else 1))

    for i in range(1, 2 ** limit):
        d[i] = {}
        # We won't use the last value as a threshold, as it won't refine anything, as everything <= maxval
        for cv in range(0, max_domain):
            d[i][cv] = pool.id(f"d{i}_{cv}")

    f = {}
    for i in range(1, 2**limit):
        f[i] = {}
        for cf in range(1, instance.num_features + 1):
            if len(instance.domains[cf]) > 0:
                f[i][cf] = pool.id(f"f{i}_{cf}")

    c_vars = len(next(iter(class_map.values())))
    c = {}
    for i in range(0, 2**limit):
        c[i] = {}
        for j in range(0, c_vars):
            c[i][j] = pool.id(f"c{i}_{j}")

    return x, f, c, d, pool


def encode(instance, limit, solver, opt_size=False, multiclass=False):
    classes = list(instance.classes)  # Give classes an order
    if opt_size:
        classes.insert(0, "EmptyLeaf")
    c_vars = len(bin(len(classes)-1)) - 2  # "easier" than log_2
    c_values = list(itertools.product([True, False], repeat=c_vars))
    class_map = {}
    for i in range(0, len(classes)):
        class_map[classes[i]] = c_values.pop()

    x, f, c, d, p = _init_var(instance, limit, class_map)

    # each node has a feature
    for i in range(1, 2**limit):
        clause = []
        for f1, f1v in f[i].items():
            clause.append(f1v)
            for f2, f2v in f[i].items():
                if f2 > f1:
                    solver.add_clause([-f1v, -f2v])
        solver.add_clause(clause)

    # each node has a threshold
    for i in range(1, 2**limit):
        clause = []
        for d1, d1v in d[i].items():
            clause.append(d1v)
            for d2, d2v in d[i].items():
                if d2 > d1:
                    solver.add_clause([-d1v, -d2v])
        solver.add_clause(clause)

    for i in range(0, len(instance.examples)):
        _alg1(instance, i, limit, 0, 1, list(), f, x, d, solver)
        _alg2(instance, i, limit, 0, 1, list(), class_map, x, c, solver)

    # Forbid non-existing classes
    for c_c in c_values:
        for c_n in range(0, 2**limit):
            clause = []
            for i in range(0, c_vars):
                clause.append(-c[c_n][i] if c_c[i] else c[c_n][i])
            solver.add_clause(clause)
    return {"f": f, "x": x, "c": c, "class_map": class_map, "pool": p, "d": d}


def _alg1(instance, e_idx, limit, lvl, q, clause, fs, x, d, solver):
    if lvl == limit:
        return

    example = instance.examples[e_idx]
    for f in range(1, instance.num_features + 1):
        if len(instance.domains[f]) == 0:
            continue

        is_cat = f in instance.is_categorical
        for i2 in range(0, len(d[q])):
            if i2 < len(instance.domains[f]) - 1:
                c_val = example.features[f] if example.features[f] != "?" else instance.domains_max[f]

                if (not is_cat and c_val > instance.domains[f][i2]) or (is_cat and c_val != instance.domains[f][i2]):
                    solver.add_clause([*clause, -x[e_idx][lvl], -d[q][i2], -fs[q][f]])
            else:
                solver.add_clause([-fs[q][f], -d[q][i2]])

    n_cl = list(clause)
    n_cl.append(-x[e_idx][lvl])
    _alg1(instance, e_idx, limit, lvl+1, 2 * q + 1, n_cl, fs, x, d, solver)

    for f in range(1, instance.num_features + 1):
        if len(instance.domains[f]) == 0:
            continue

        is_cat = f in instance.is_categorical
        for i2 in range(0, len(d[q])):
            if i2 < len(instance.domains[f]) - 1:
                c_val = example.features[f] if example.features[f] != "?" else instance.domains_max[f]
                if (not is_cat and c_val <= instance.domains[f][i2]) or (is_cat and c_val == instance.domains[f][i2]):
                    solver.add_clause([*clause, x[e_idx][lvl], -d[q][i2], -fs[q][f]])

    n_cl2 = list(clause)
    n_cl2.append(x[e_idx][lvl])
    _alg1(instance, e_idx, limit, lvl+1, 2*q, n_cl2, fs, x, d, solver)


def _alg2(instance, e_idx, limit, lvl, q, clause, class_map, x, c, solver):
    if lvl == limit:
        c_vars = class_map[instance.examples[e_idx].cls]
        for i in range(0, len(c_vars)):
            if c_vars[i]:
                solver.add_clause([*clause, c[q - 2 ** limit][i]])
            else:
                solver.add_clause([*clause, -c[q - 2 ** limit][i]])
    else:
        n_cl = list(clause)
        n_cl.append(x[e_idx][lvl])
        n_cl2 = list(clause)
        n_cl2.append(-x[e_idx][lvl])
        _alg2(instance, e_idx, limit, lvl+1, 2*q, n_cl, class_map, x, c, solver)
        _alg2(instance, e_idx, limit, lvl+1, 2*q+1, n_cl2, class_map, x, c, solver)


def encode_size(vs, instance, solver, dl):
    pool = vs["pool"]
    card_vars = []
    c = vs["c"]

    for c_n in range(0, 2 ** dl):
        card_vars.append(pool.id(f"n{c_n}"))
        for c_c in c[c_n].values():
            solver.add_clause([-c_c, pool.id(f"n{c_n}")])

    return card_vars


def estimate_size_add(instance, dl):
    c = len(instance.classes)
    return 2 ** dl * c * 2 + (2 ** dl) ** 2 * 3


def encode_extended_leaf_size(vs, instance, solver, dl):
    pool = vs["pool"]
    card_vars = []
    c = vs["c"]
    cm = vs["class_map"]

    for c_n in range(0, 2 ** dl):
        for cls, vals in cm.items():
            if not cls.startswith("-"):
                clause = []
                for i in range(0, len(vals)):
                    clause.append(-c[c_n][i] if vals[i] else c[c_n][i])
                clause.append(pool.id(f"e{c_n}"))
                solver.add_clause(clause)

        card_vars.append(pool.id(f"e{c_n}"))
    return card_vars


def encode_extended_leaf_limit(vs, solver, dl):
    c = vs["c"]
    cm = vs["class_map"]

    for cls, vals in cm.items():
        if not cls.startswith("-") and cls != "EmptyLeaf":
            for c_n in range(0, 2 ** dl):
                for c_n2 in range(c_n+1, 2 ** dl):
                    clause = []
                    for i in range(0, len(vals)):
                        clause.append(-c[c_n][i] if vals[i] else c[c_n][i])
                        clause.append(-c[c_n2][i] if vals[i] else c[c_n2][i])
                    solver.add_clause(clause)


def _decode(model, instance, limit, vs):
    class_map = vs['class_map']
    fs = vs["f"]
    cs = vs["c"]
    ds = vs["d"]

    num_leafs = 2**limit
    tree = DecisionTree()

    # Find features
    for i in range(1, num_leafs):
        f_found = False
        for f, fv in fs[i].items():
            if model[fv]:
                if f_found:
                    print(f"ERROR: double features found for node {i}, features {f} and {tree.nodes[i].feature}")
                f_found = True
                threshold_id = None
                for d, dv in ds[i].items():
                    if model[dv]:
                        threshold_id = d

                tsh = instance.domains[f][threshold_id]
                if i == 1:
                    tree.set_root(f, tsh, f in instance.is_categorical)
                else:
                    tree.add_node(f, tsh, i // 2, i % 2 == 1, f in instance.is_categorical)
        if not f_found:
            print(f"ERROR: No feature found for node {i}")

    for c_c, c_v in class_map.items():
        for i in range(0, num_leafs):
            all_right = True
            for i_v in range(0, len(c_v)):
                if bool(model[cs[i][i_v]]) != c_v[i_v]:
                    all_right = False
                    break
            if all_right:
                if num_leafs == 1:
                    tree.set_root_leaf(c_c)
                else:
                    tree.add_leaf(c_c, (num_leafs + i) // 2, i % 2 == 1)

    #_reduce_tree(tree, instance)
    tree.clean(instance)
    return tree


def check_consistency(self, model, instance, num_nodes, tree):
    pass


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


def estimate_size(instance, depth):
    """Estimates the required size in the number of literals"""

    d2 = 2**depth
    c = len(instance.classes)
    lc = len(bin(c-1)) - 2  #ln(c)
    s = len(instance.examples)
    f = sum(len(x) for x in instance.domains)

    forbidden_c = (2**lc - c) * d2 * lc
    alg1_lits = s * sum(2**i * f * (i+2) for i in range(0, depth))

    return d2 * f * (f-1) // 2 + d2 * f + forbidden_c + alg1_lits + s * d2 * (depth+1) * lc


def is_sat():
    return True