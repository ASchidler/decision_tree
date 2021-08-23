import itertools
from sys import maxsize, stdout
from threading import Timer
from decimal import Decimal
from pysat.formula import IDPool
from pysat.card import ITotalizer
from nonbinary.decision_tree import DecisionTreeNode, DecisionTreeLeaf, DecisionTree
import time

import z3


def _init_var(instance, limit, slv, opt_size=False):
    x = {}
    for xl in range(0, len(instance.examples)):
        x[xl] = {}
        for x2 in range(0, limit):
            x[xl][x2] = z3.Bool(f"x{xl}_{x2}")

    f = {}
    for i in range(1, 2**limit):
        f[i] = z3.Int(f"f{i}")
        slv.add(f[i] > 0)
        slv.add(f[i] <= instance.num_features)

    t = {}
    for i in range(1, 2 ** limit):
        t[i] = z3.Real(f"t{i}")

    c = {}
    for i in range(0, 2**limit):
        c[i] = z3.Int(f"c{i}")
        if opt_size:
            slv.add(c[i] >= 0)
        else:
            slv.add(c[i] > 0)
        slv.add(c[i] <= len(instance.classes))

    return x, f, c, t


def encode(instance, limit, slv, opt_size=False):
    classes = list(instance.classes)  # Give classes an order
    classes.insert(0, "EmptyLeaf")
    class_map = {cc: i for i, cc in enumerate(classes)}

    x, f, c, t = _init_var(instance, limit, slv)

    for i in range(0, len(instance.examples)):
        _alg1(instance, i, limit, 0, 1, list(), f, x, slv, t)
        _alg2(instance, i, limit, 0, 1, list(), class_map, x, c, slv)

    return {"f": f, "x": x, "c": c, "class_map": class_map, "t": t, "classes": classes}


def _alg1(instance, e_idx, limit, lvl, q, clause, fs, x, slv, t):
    if lvl == limit:
        return

    example = instance.examples[e_idx]
    for f in range(1, instance.num_features + 1):
        c_val = example.features[f] if example.features[f] != "?" else instance.domains_max[f]
        if f in instance.is_categorical:
            for c_i in range(0, len(instance.domains[f])):
                if instance.domains[f][c_i] != c_val:
                    slv.add(z3.Or([*clause, z3.Not(x[e_idx][lvl]), c_i != t[q], fs[q] != f]))
                else:
                    slv.add(z3.Or([*clause, z3.Not(x[e_idx][lvl]), c_i == t[q], fs[q] != f]))
        else:
            slv.add(z3.Or([*clause, z3.Not(x[e_idx][lvl]), c_val <= t[q], fs[q] != f]))

    n_cl = list(clause)
    n_cl.append(z3.Not(x[e_idx][lvl]))
    _alg1(instance, e_idx, limit, lvl+1, 2 * q + 1, n_cl, fs, x, slv, t)

    for f in range(1, instance.num_features + 1):
        c_val = example.features[f] if example.features[f] != "?" else instance.domains_max[f]
        if f in instance.is_categorical:
            for c_i in range(0, len(instance.domains[f])):
                if instance.domains[f][c_i] == c_val:
                    slv.add(z3.Or([*clause, x[e_idx][lvl], c_i != t[q], fs[q] != f]))
        else:
            slv.add(z3.Or([*clause, x[e_idx][lvl], c_val > t[q], fs[q] != f]))

    n_cl2 = list(clause)
    n_cl2.append(x[e_idx][lvl])
    _alg1(instance, e_idx, limit, lvl+1, 2*q, n_cl2, fs, x, slv, t)


def _alg2(instance, e_idx, limit, lvl, q, clause, class_map, x, c, slv):
    if lvl == limit:
        slv.add(z3.Or([*clause, c[q-2 ** limit] == class_map[instance.examples[e_idx].cls]]))
    else:
        n_cl = list(clause)
        n_cl.append(x[e_idx][lvl])
        n_cl2 = list(clause)
        n_cl2.append(z3.Not(x[e_idx][lvl]))
        _alg2(instance, e_idx, limit, lvl+1, 2*q, n_cl, class_map, x, c, slv)
        _alg2(instance, e_idx, limit, lvl+1, 2*q+1, n_cl2, class_map, x, c, slv)


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
            for cc, c_vars in class_map:
                n_n = z3.Int(f"n{c_n}")
                card_vars.append(n_n)
                clause = []

                for i in range(0, len(c_vars)):
                    clause.append(c[c_n][i] if c_vars[i] else z3.Not(c[c_n][i]))
                solver.add(z3.Implies(z3.And(clause), n_n == (instance.class_sizes[cc] if cc in instance.class_sizes[cc] else 1)))

    return card_vars


def estimate_size_add(instance, dl):
    c = len(instance.classes)
    return 2 ** dl * c * 2 + (2 ** dl) ** 2 * 3


def run(instance, start_bound=1, ub=maxsize, timeout=0, opt_size=False, check_mem=True, slim=True, multiclass=False):
    c_bound = start_bound
    c_lb = 1
    best_model = None
    best_depth = None

    c_start = time.time()

    while c_lb < ub:
        print(f"Running depth {c_bound}")
        stdout.flush()

        slv = z3.Solver()
        slv.set("max_memory", 10000)

        vs = encode(instance, c_bound, slv, opt_size=opt_size)
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
        if model:
            best_model = _decode(model, instance, best_depth, vs)

    return best_model


def _decode(model, instance, limit, vs):
    classes = vs['classes']
    fs = vs["f"]
    cs = vs["c"]
    ts = vs["t"]

    num_leafs = 2**limit
    tree = DecisionTree()

    # Find features
    for i in range(1, num_leafs):
        f = int(model[fs[i]].as_long())
        threshold = model[ts[i]]
        if threshold is None:  # Edge case where the threshold is not needed
            if f in instance.is_categorical:
                threshold = 0
            else:
                threshold = instance.domains[f][0]
        else:
            threshold = threshold.as_decimal(6)

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

    for i in range(0, num_leafs):
        c_c = classes[int(model[cs[i]].as_long())]
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
    lc = len(bin(c-1)) - 2  #ln(c)
    s = len(instance.examples)
    f = sum(len(x) for x in instance.domains)

    forbidden_c = (2**lc - c) * d2 * lc
    alg1_lits = s * sum(2**i * f * (i+2) for i in range(0, depth))

    return d2 * f * (f-1) // 2 + d2 * f + forbidden_c + alg1_lits + s * d2 * (depth+1) * lc


def is_sat():
    return False
