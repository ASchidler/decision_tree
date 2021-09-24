import itertools
from sys import maxsize, stdout
from threading import Timer
from collections import defaultdict
from decimal import Decimal
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

    c = {}
    for i in range(0, 2**limit):
        c[i] = {}
        if len(class_map) <= 2:
            c[i] = z3.Bool(f"c{i}")
        else:
            for j in range(1, len(class_map)+1):
                c[i][j] = z3.Bool(f"c{i}_{j}")

    return x, f, c, t


def encode(instance, limit, slv, opt_size=False):
    classes = list(instance.classes)  # Give classes an order

    if opt_size:
        classes.insert(0, "EmptyLeaf")

    if len(classes) == 2 or opt_size:
        class_map = {cc: i for i, cc in enumerate(classes)}
    else:
        class_map = {cc: i + 1 for i, cc in enumerate(classes)}

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

    if len(class_map) > 2:
        for c_n in range(0, 2 ** limit):
            clause = []

            for c1, c1v in c[c_n].items():
                clause.append(c1v)
                for c2, c2v in c[c_n].items():
                    if c2 > c1:
                        slv.add(z3.Or([z3.Not(c1v), z3.Not(c2v)]))

            if not opt_size:
                slv.add(z3.Or(clause))

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

        if len(class_map) <= 2:
            if c_vars == 0:
                slv.add(z3.Or([*clause, z3.Not(c[q - 2**limit])]))
            else:
                slv.add(z3.Or([*clause, c[q - 2 ** limit]]))
        else:
            slv.add(z3.Or([*clause, c[q - 2 ** limit][c_vars]]))
    else:
        clause.append(x[e_idx][lvl])
        _alg2(instance, e_idx, limit, lvl + 1, 2 * q, clause, class_map, x, c, slv)
        clause.pop()

        clause.append(z3.Not(x[e_idx][lvl]))
        _alg2(instance, e_idx, limit, lvl+1, 2*q+1, clause, class_map, x, c, slv)
        clause.pop()


def encode_size(vs, instance, solver, dl):
    card_vars = []
    c = vs["c"]

    for c_n in range(0, 2 ** dl):
        n_n = z3.Int(f"n{c_n}")
        card_vars.append(n_n)
        for c_c in c[c_n].values():
            solver.add(z3.Implies(c_c, n_n == 1))

    return card_vars


def encode_size_surrogate(vs, instance, solver, dl):
    card_vars = []
    class_map = vs["class_map"]
    c = vs["c"]

    # If there are no virtual classes, no need for this
    if instance.class_sizes is None or len(instance.class_sizes):
        encode_size(vs, instance, solver, dl)
    else:
        for c_n in range(0, 2 ** dl):
            for cc, c_vars in class_map.items():
                if cc == "EmptyLeaf":
                    continue

                n_n = z3.Int(f"n{c_n}")
                card_vars.append(n_n)

                solver.add(z3.Implies(c_vars, n_n == (instance.class_sizes[cc] if cc in instance.class_sizes[cc] else 1)))

    return card_vars


def encode_extended_leaf_limit(vs, solver, dl):
    c = vs["c"]
    cm = vs["class_map"]

    for cls, vals in cm.items():
        if not cls.startswith("-") and cls != "EmptyLeaf":
            for c_n in range(0, 2 ** dl):
                for c_n2 in range(c_n + 1, 2 ** dl):
                    solver.add([z3.Not(c[c_n][vals]), z3.Not(c[c_n2][vals])])


def estimate_size_add(instance, dl):
    c = len(instance.classes)
    return 2 ** dl * c * 2 + (2 ** dl) ** 2 * 3


def run(instance, start_bound=1, ub=maxsize, timeout=0, opt_size=False, check_mem=True, slim=True, maintain=False, limit_size=0):
    c_bound = start_bound
    c_lb = 1
    best_model = None
    best_depth = None

    # Edge cases
    if len(instance.classes) == 1:
        dt = DecisionTree()
        dt.set_root_leaf(next(iter(instance.classes)))
        return dt

    if all(len(x) == 0 for x in instance.domains):
        counts = defaultdict(int)
        for e in instance.examples:
            counts[e.cls] += 1
        _, cls = max((v, k) for k, v in counts.items())
        dt = DecisionTree()
        dt.set_root_leaf(cls)
        return dt

    c_start = time.time()

    while c_lb < ub:
        print(f"Running depth {c_bound}")
        stdout.flush()

        slv = z3.Solver()
        slv.set("max_memory", 10000)

        vs = encode(instance, c_bound, slv, opt_size=opt_size or maintain)
        if limit_size > 0 and maintain:
            encode_extended_leaf_limit(vs, slv, c_bound)
            cards = encode_size(vs, instance, slv, best_depth)
            slv.add(z3.Sum(cards) <= {limit_size})
        if timeout > 0:
            if (time.time() - c_start) > timeout:
                break

            slv.set("timeout", int(timeout - (time.time() - c_start)) * 1000)
        res = slv.check()

        if res == z3.sat:
            model = slv.model()
            best_model = _decode(model, instance, c_bound, vs)
            best_depth = best_model.get_depth()
            ub = best_depth
            c_bound = ub - 1
        else:
            c_bound += 1
            c_lb = c_bound

    c_start = time.time()
    if opt_size and best_model:
        opt = z3.Optimize()
        # Not supported by optimize
        # opt.set("max_memory", 10000)
        vs = encode(instance, best_depth, opt)
        if timeout > 0:
            if (time.time() - c_start) > timeout:
                return best_model

            opt.set("timeout", int(timeout - (time.time() - c_start)) * 1000)
        cards = encode_size_surrogate(vs, instance, opt, best_depth) if slim else encode_size(vs, instance, opt, best_depth)

        c_size_bound = best_model.root.get_leaves() - 1
        c_opt = z3.Int("c_opt")
        opt.add(z3.Sum(cards) <= c_opt)
        opt.add(c_opt <= c_size_bound)
        opt.minimize(c_opt)
        opt.check()

        model = opt.model()
        if len(model) > 0:
            best_model = _decode(model, instance, best_depth, vs)

    return best_model


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
                threshold = model[ts[i]].as_decimal(6)
                if threshold.endswith("?"):
                    threshold = threshold[:-1]

                if f_found:
                    print(f"ERROR: double features found for node {i}, features {f} and {tree.nodes[i].feature}")
                else:
                    if f in instance.is_categorical:
                        threshold = float(threshold)
                        if threshold.is_integer() and threshold < len(instance.domains[f]):
                            threshold = instance.domains[f][int(threshold)]
                    else:
                        threshold = Decimal(threshold)

                    if i == 1:
                        tree.set_root(f, threshold, f in instance.is_categorical)
                    else:
                        tree.add_node(f, threshold, i//2, i % 2 == 1, f in instance.is_categorical)

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

    s = len(instance.examples)
    f = sum(len(instance.domains[x]) for x in range(1, instance.num_features+1))

    alg1_lits = s * sum(2**i * f * (i+2) for i in range(0, depth))

    return d2 * f * (f-1) // 2 + d2 * f + alg1_lits + s * d2 * (depth+1) * c


def is_sat():
    return False


def run_incremental(strategy, increment=1, timeout=300, opt_size=False):
    c_bound = 1
    best_model = None

    strategy.find_next(1+increment)

    c_start = time.time()

    while (time.time() - c_start) < timeout:
        solved = False
        # Edge cases
        if len(strategy.get_instance().classes) == 1 and not solved:
            best_model = DecisionTree()
            best_model.set_root_leaf(next(iter(strategy.get_instance().classes)))
            solved = True

        if all(len(x) == 0 for x in strategy.get_instance().domains) and not solved:
            counts = defaultdict(int)
            for e in strategy.get_instance().examples:
                counts[e.cls] += 1
            _, cls = max((v, k) for k, v in counts.items())
            best_model = DecisionTree()
            best_model.set_root_leaf(cls)
            solved = True

        if not solved:
            print(f"Running {len(strategy.get_instance().examples)} / {c_bound}")
            stdout.flush()

            slv = z3.Solver()
            slv.set("max_memory", 10000)

            vs = encode(strategy.get_instance(), c_bound, slv)
            slv.set("timeout", int(timeout - (time.time() - c_start)) * 1000)
            res = slv.check()

            if res == z3.sat:
                model = slv.model()
                best_model = _decode(model, strategy.get_instance(), c_bound, vs)
                strategy.unreduce(best_model)
                solved = True
            else:
                c_bound += 1

        if strategy.done():
            break

        if solved:
            strategy.find_next(increment)

    return best_model


def get_tree_size(tree):
    return tree.root.get_leaves()
