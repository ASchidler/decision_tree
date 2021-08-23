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

    f = {}
    for i in range(1, 2**limit):
        f[i] = {}
        for cf in range(1, instance.num_features + 1):
            if len(instance.domains[cf]) == 0:
                continue

            # We don't need an entry for the last variable, as <= maxval is redundant
            for j in range(0, len(instance.domains[cf]) - (0 if cf in instance.is_categorical else 1)):
                f[i][instance.feature_idx[cf] + j] = pool.id(f"f{i}_{instance.feature_idx[cf] + j}")

    c_vars = len(next(iter(class_map.values())))
    c = {}
    for i in range(0, 2**limit):
        c[i] = {}
        for j in range(0, c_vars):
            c[i][j] = pool.id(f"c{i}_{j}")

    return x, f, c, pool


def encode(instance, limit, solver, opt_size=False, multiclass=False):
    classes = list(instance.classes)  # Give classes an order
    if opt_size:
        classes.insert(0, "EmptyLeaf")
    c_vars = len(bin(len(classes)-1)) - 2  # "easier" than log_2
    c_values = list(itertools.product([True, False], repeat=c_vars))
    class_map = {}
    for i in range(0, len(classes)):
        class_map[classes[i]] = c_values.pop()

    x, f, c, p = _init_var(instance, limit, class_map)

    # each node has a feature
    for i in range(1, 2**limit):
        clause = []

        for f1, f1v in f[i].items():
            clause.append(f1v)
            for f2, f2v in f[i].items():
                if f2 > f1:
                    solver.add_clause([-f1v, -f2v])
        solver.add_clause(clause)

    for i in range(0, len(instance.examples)):
        _alg1(instance, i, limit, 0, 1, list(), f, x, solver)
        _alg2(instance, i, limit, 0, 1, list(), class_map, x, c, solver, multiclass)

    # Forbid non-existing classes
    for c_c in c_values:
        for c_n in range(0, 2**limit):
            clause = []
            for i in range(0, c_vars):
                clause.append(-c[c_n][i] if c_c[i] else c[c_n][i])
            solver.add_clause(clause)
    return {"f": f, "x": x, "c": c, "class_map": class_map, "pool": p}


def _alg1(instance, e_idx, limit, lvl, q, clause, fs, x, solver):
    if lvl == limit:
        return

    example = instance.examples[e_idx]
    for f in range(1, instance.num_features + 1):
        if len(instance.domains[f]) == 0:
            continue

        base_idx = instance.feature_idx[f]
        is_cat = f in instance.is_categorical
        for i2 in range(0, len(instance.domains[f]) - (0 if f in instance.is_categorical else 1)):
            c_val = example.features[f] if example.features[f] != "?" else instance.domains_max[f]
            if (not is_cat and c_val > instance.domains[f][i2]) or (is_cat and c_val != instance.domains[f][i2]):
                solver.add_clause([*clause, -x[e_idx][lvl], -fs[q][base_idx + i2]])
            if (not is_cat and c_val <= instance.domains[f][i2]) or (is_cat and c_val == instance.domains[f][i2]):
                solver.add_clause([*clause, x[e_idx][lvl], -fs[q][base_idx + i2]])

    clause.append(-x[e_idx][lvl])
    _alg1(instance, e_idx, limit, lvl+1, 2 * q + 1, clause, fs, x, solver)
    clause.pop()

    clause.append(x[e_idx][lvl])
    _alg1(instance, e_idx, limit, lvl+1, 2*q, clause, fs, x, solver)
    clause.pop()


def _alg2(instance, e_idx, limit, lvl, q, clause, class_map, x, c, solver, multiclass=False):
    if lvl == limit:
        c_vars = class_map[instance.examples[e_idx].cls]
        c_vars2 = None
        if instance.examples[e_idx].surrogate_cls:
            c_vars2 = class_map[instance.examples[e_idx].surrogate_cls]

        for i in range(0, len(c_vars)):
            cv1 = c[q - 2 ** limit][i] * (1 if c_vars[i] else -1)
            if c_vars2 is None or not multiclass or c_vars[i] == c_vars2[i]:
                solver.add_clause([*clause, cv1])
            else:
                for j in range(0, len(c_vars)):
                    if c_vars[j] != c_vars2[j]:
                        cv2 = c[q - 2 ** limit][j] * (1 if c_vars[j] else -1)
                        solver.add_clause([*clause, cv1, cv2])
    else:
        clause.append(x[e_idx][lvl])
        _alg2(instance, e_idx, limit, lvl + 1, 2 * q, clause, class_map, x, c, solver, multiclass)
        clause.pop()

        clause.append(-x[e_idx][lvl])
        _alg2(instance, e_idx, limit, lvl+1, 2*q+1, clause, class_map, x, c, solver, multiclass)
        clause.pop()


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


def estimate_size_add(instance, dl):
    c = len(instance.classes)
    return 2 ** dl * c * 2 + (2 ** dl) ** 2 * 3


def _decode(model, instance, limit, vs):
    class_map = vs['class_map']
    fs = vs["f"]
    cs = vs["c"]

    num_leafs = 2**limit
    tree = DecisionTree()

    # Find features
    for i in range(1, num_leafs):
        f_found = False
        for f, fv in fs[i].items():
            if model[fv]:
                if f_found:
                    raise RuntimeError(f"ERROR: double features found for node {i}, features {f} and {tree.nodes[i].feature}")
                f_found = True
                real_f = None

                if instance.num_features == 1:
                    real_f = 1
                    tsh = instance.domains[1][f - instance.feature_idx[1]]
                else:
                    for r_f in range(2, instance.num_features+1):
                        real_f = None
                        if f >= instance.feature_idx[r_f - 1] and f < instance.feature_idx[r_f]:
                            real_f = r_f - 1
                        elif r_f == instance.num_features:
                            real_f = r_f
                        if real_f is not None:
                            tsh = instance.domains[real_f][f - instance.feature_idx[real_f]]
                            break

                if i == 1:
                    tree.set_root(real_f, tsh, real_f in instance.is_categorical)
                else:
                    tree.add_node(real_f, tsh, i // 2, i % 2 == 1, real_f in instance.is_categorical)

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
    lc = len(bin(c-1)) - 2  #ln(c)
    s = len(instance.examples)
    f = sum(len(x) for x in instance.domains)

    forbidden_c = (2**lc - c) * d2 * lc
    alg1_lits = s * sum(2**i * f * (i+2) for i in range(0, depth))

    return d2 * f * (f-1) // 2 + d2 * f + forbidden_c + alg1_lits + s * d2 * (depth+1) * lc
