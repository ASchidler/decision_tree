import itertools
from sys import maxsize, stdout
from threading import Timer

from pysat.formula import IDPool
from pysat.card import ITotalizer
from nonbinary.decision_tree import DecisionTreeNode, DecisionTreeLeaf, DecisionTree
import time

import z3


def _init_var(instance, limit, class_map):
    x = {}
    for xl in range(0, len(instance.examples)):
        x[xl] = {}
        for x2 in range(0, limit):
            x[xl][x2] = z3.Bool(f"x{xl}_{x2}")

    f = {}
    for i in range(1, 2**limit):
        f[i] = {}
        for j in range(1, instance.num_features + 1):
            f[i][j] = z3.Bool(f"f{i}_{j}")

    t = {}
    for i in range(1, 2 ** limit):
        t[i] = z3.Real(f"t{i}")

    c_vars = len(next(iter(class_map.values())))
    c = {}
    for i in range(0, 2**limit):
        c[i] = {}
        for j in range(0, c_vars):
            c[i][j] = z3.Bool(f"c{i}_{j}")

    return x, f, c, t


def encode(instance, limit, slv, opt_size=False):
    classes = list(instance.classes)  # Give classes an order

    if opt_size:
        classes.insert(0, "EmptyLeaf")
    c_vars = len(bin(len(classes)-1)) - 2  # "easier" than log_2
    c_values = list(itertools.product([True, False], repeat=c_vars))
    class_map = {}
    for i in range(0, len(classes)):
        class_map[classes[i]] = c_values.pop()

    x, f, c, t = _init_var(instance, limit, class_map)

    # each node has a feature
    for i in range(1, 2**limit):
        clause = []
        for f1, f1v in f[i].items():
            clause.append(f1v)
            for f2, f2v in f[i].items():
                if f2 > f1:
                    slv.add(z3.Or([z3.Not(f1v), z3.Not(f2v)]))

            # In case of binary features, we can fix the threshold to the middle
            # if instance.is_binary[f1]:
            #     middle = round(sum(instance.domains[f1]) / 2, 1)
            #     slv.add(z3.Or([z3.Not(f[i][f1]), t[i] == middle]))

        slv.add(z3.Or(clause))

    for i in range(0, len(instance.examples)):
        _alg1(instance, i, limit, 0, 1, list(), f, x, slv, t)
        _alg2(instance, i, limit, 0, 1, list(), class_map, x, c, slv)

    # Forbid non-existing classes
    # for c_c in c_values:
    #     for c_n in range(0, 2**limit):
    #         clause = []
    #         for i in range(0, c_vars):
    #             clause.append(z3.Not(c[c_n][i]) if c_c[i] else c[c_n][i])
    #         slv.add(clause)
    return {"f": f, "x": x, "c": c, "class_map": class_map, "t": t}


def _alg1(instance, e_idx, limit, lvl, q, clause, fs, x, slv, t):
    if lvl == limit:
        return

    example = instance.examples[e_idx]
    for f in range(1, instance.num_features + 1):
        c_val = example.features[f] if example.features[f] != "?" else instance.domains_max[f]
        if f in instance.is_categorical:
            for c_i in range(0, len(instance.domains[f])):
                if instance.domains[f][c_i] != c_val:
                    slv.add(z3.Or([*clause, z3.Not(x[e_idx][lvl]), c_i != t[q], z3.Not(fs[q][f])]))
                else:
                    slv.add(z3.Or([*clause, z3.Not(x[e_idx][lvl]), c_i == t[q], z3.Not(fs[q][f])]))
        else:
            slv.add(z3.Or([*clause, z3.Not(x[e_idx][lvl]), c_val <= t[q], z3.Not(fs[q][f])]))

    n_cl = list(clause)
    n_cl.append(z3.Not(x[e_idx][lvl]))
    _alg1(instance, e_idx, limit, lvl+1, 2 * q + 1, n_cl, fs, x, slv, t)

    for f in range(1, instance.num_features + 1):
        c_val = example.features[f] if example.features[f] != "?" else instance.domains_max[f]
        if f in instance.is_categorical:
            for c_i in range(0, len(instance.domains[f])):
                if instance.domains[f][c_i] == c_val:
                    slv.add(z3.Or([*clause, x[e_idx][lvl], c_i != t[q], z3.Not(fs[q][f])]))
        else:
            slv.add(z3.Or([*clause, x[e_idx][lvl], c_val > t[q], z3.Not(fs[q][f])]))

    n_cl2 = list(clause)
    n_cl2.append(x[e_idx][lvl])
    _alg1(instance, e_idx, limit, lvl+1, 2*q, n_cl2, fs, x, slv, t)


def _alg2(instance, e_idx, limit, lvl, q, clause, class_map, x, c, slv):
    if lvl == limit:
        c_vars = class_map[instance.examples[e_idx].cls]
        for i in range(0, len(c_vars)):
            if c_vars[i]:
                slv.add(z3.Or([*clause, c[q - 2 ** limit][i]]))
            else:
                slv.add(z3.Or([*clause, z3.Not(c[q - 2 ** limit][i])]))
    else:
        n_cl = list(clause)
        n_cl.append(x[e_idx][lvl])
        n_cl2 = list(clause)
        n_cl2.append(z3.Not(x[e_idx][lvl]))
        _alg2(instance, e_idx, limit, lvl+1, 2*q, n_cl, class_map, x, c, slv)
        _alg2(instance, e_idx, limit, lvl+1, 2*q+1, n_cl2, class_map, x, c, slv)


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


def run(instance, start_bound=1, ub=maxsize, opt_size=False):
    c_bound = start_bound
    c_lb = 1
    best_model = None
    best_depth = None

    while c_lb < ub:
        print(f"Running depth {c_bound}")
        stdout.flush()
        #with z3.Solver() as slv:
        slv = z3.Solver()
        vs = encode(instance, c_bound, slv)
        res = slv.check()

        if res == z3.sat:
            model = slv.model()
            best_model = _decode(model, instance, c_bound, vs)
            best_depth = c_bound
            ub = c_bound
            c_bound -= 1
        else:
            c_bound += 1
            c_lb = c_bound
    #
    # if opt_size and best_model:
    #     with solver() as slv:
    #         c_size_bound = best_model.root.get_leafs() - 1
    #         solved = True
    #         vs = encode(instance, best_depth, slv)
    #         card = encode_size(vs, instance, slv, best_depth)
    #
    #         tot = ITotalizer(card, c_size_bound, top_id=vs["pool"].top+1)
    #         slv.append_formula(tot.cnf)
    #
    #         while solved:
    #             print(f"Running size {c_size_bound}")
    #             stdout.flush()
    #             if timeout == 0:
    #                 solved = slv.solve()
    #             else:
    #                 def interrupt(s):
    #                     s.interrupt()
    #
    #                 timer = Timer(timeout, interrupt, [slv])
    #                 timer.start()
    #                 solved = slv.solve_limited(expect_interrupt=True)
    #                 timer.cancel()
    #
    #             if solved:
    #                 model = {abs(x): x > 0 for x in slv.get_model()}
    #                 best_model = _decode(model, instance, best_depth, vs)
    #                 c_size_bound -= 1
    #                 slv.add_clause([-tot.rhs[c_size_bound]])
    #             else:
    #                 break

    return best_model


def extend(slv, instance, vs, c_bound, increment, size_limit):
    c = len(instance.classes)
    lc = len(bin(c - 1)) - 2  # ln(c)
    f = instance.num_features
    d2 = 2 ** c_bound

    alg1_lits = increment * sum(2 ** i * f * (i + 2) for i in range(0, c_bound))
    guess = alg1_lits + increment * d2 * (c_bound + 1) * lc

    if guess > size_limit:
        return None

    for e_idx in range(len(instance.examples) - increment, len(instance.examples)):
        vs["x"][e_idx] = {}
        for x2 in range(0, c_bound):
            vs["x"][e_idx][x2] = vs["pool"].id(f"x{e_idx}{x2}")
        _alg1(instance, e_idx, c_bound, 0, 1, list(), vs["f"], vs["x"], slv)
        _alg2(instance, e_idx, c_bound, 0, 1, list(), vs["class_map"], vs["x"], vs["c"], slv)

    return guess


def _decode(model, instance, limit, vs):
    class_map = vs['class_map']
    fs = vs["f"]
    cs = vs["c"]
    ts = vs["t"]

    num_leafs = 2**limit
    tree = DecisionTree()

    # Find features
    for i in range(1, num_leafs):
        f_found = False
        for f in range(1, instance.num_features+1):
            if model[fs[i][f]]:
                threshold = model[ts[i]].as_decimal(5)
                if threshold.endswith("?"):
                    threshold = threshold[:-1]
                threshold = float(threshold)
                if f_found:
                    print(f"ERROR: double features found for node {i}, features {f} and {tree.nodes[i].feature}")
                else:
                    if f in instance.is_categorical:
                        if threshold.is_integer() and threshold < len(instance.domains[f]):
                            threshold = instance.domains[f][int(threshold)]
                    if i == 1:
                        tree.set_root(f, threshold, f in instance.is_categorical)
                    else:
                        tree.add_node(f, threshold, i//2, i % 2 == 1, f in instance.is_categorical)

    for c_c, c_v in class_map.items():
        for i in range(0, num_leafs):
            all_right = True
            for i_v in range(0, len(c_v)):
                if bool(model[cs[i][i_v]]) != c_v[i_v]:
                    all_right = False
                    break
            if all_right:
                tree.add_leaf(c_c, (num_leafs + i)//2, i % 2 == 1)

    #_reduce_tree(tree, instance)
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
