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
    e = {}
    for i in range(1, 2**limit):
        f[i] = {}
        e[i] = pool.id(f"e{i}")
        for cf in range(1, instance.num_features + 1):
            # We don't need an entry for the last variable, as <= maxval is redundant
            for j in range(0, len(instance.domains[cf]) - (0 if cf in instance.is_categorical else 1)):
                f[i][instance.feature_idx[cf] + j] = pool.id(f"f{i}_{instance.feature_idx[cf] + j}")

    c_vars = len(next(iter(class_map.values())))
    c = {}
    for i in range(0, 2**limit):
        c[i] = {}
        for j in range(0, c_vars):
            c[i][j] = pool.id(f"c{i}_{j}")

    return x, f, c, e, pool


def encode(instance, limit, solver, opt_size=False):
    classes = list(instance.classes)  # Give classes an order
    if opt_size:
        classes.insert(0, "EmptyLeaf")
    c_vars = len(bin(len(classes)-1)) - 2  # "easier" than log_2
    c_values = list(itertools.product([True, False], repeat=c_vars))
    class_map = {}
    for i in range(0, len(classes)):
        class_map[classes[i]] = c_values.pop()

    x, f, c, e, p = _init_var(instance, limit, class_map)

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
        _alg1(instance, i, limit, 0, 1, list(), f, x, e, solver)
        _alg2(instance, i, limit, 0, 1, list(), class_map, x, c, solver)

    # Forbid non-existing classes
    for c_c in c_values:
        for c_n in range(0, 2**limit):
            clause = []
            for i in range(0, c_vars):
                clause.append(-c[c_n][i] if c_c[i] else c[c_n][i])
            solver.add_clause(clause)
    return {"f": f, "x": x, "c": c, "class_map": class_map, "pool": p, "e": e}


def _alg1(instance, e_idx, limit, lvl, q, clause, fs, x, e, solver):
    if lvl == limit:
        return

    example = instance.examples[e_idx]
    for f in range(1, instance.num_features + 1):
        base_idx = instance.feature_idx[f]
        is_cat = f in instance.is_categorical
        for i2 in range(0, len(instance.domains[f]) - (0 if f in instance.is_categorical else 1)):
            if example.features[f] != "?":
                if (not is_cat and example.features[f] > instance.domains[f][i2]) or (is_cat and example.features[f] != instance.domains[f][i2]):
                    solver.add_clause([*clause, -x[e_idx][lvl], e[q], -fs[q][base_idx + i2]])
                if example.features[f] != instance.domains[f][i2]:
                    solver.add_clause([*clause, -x[e_idx][lvl], -e[q], -fs[q][base_idx + i2]])
    n_cl = list(clause)
    n_cl.append(-x[e_idx][lvl])
    _alg1(instance, e_idx, limit, lvl+1, 2 * q + 1, n_cl, fs, x, e, solver)

    for f in range(1, instance.num_features + 1):
        base_idx = instance.feature_idx[f]
        is_cat = f in instance.is_categorical
        for i2 in range(0, len(instance.domains[f]) - (0 if f in instance.is_categorical else 1)):
            if example.features[f] != "?":
                if (not is_cat and example.features[f] <= instance.domains[f][i2]) or (is_cat and example.features[f] == instance.domains[f][i2]):
                    solver.add_clause([*clause, x[e_idx][lvl], e[q], -fs[q][base_idx + i2]])
                if example.features[f] == instance.domains[f][i2]:
                    solver.add_clause([*clause, x[e_idx][lvl], -e[q], -fs[q][base_idx + i2]])
    n_cl2 = list(clause)
    n_cl2.append(x[e_idx][lvl])
    _alg1(instance, e_idx, limit, lvl+1, 2*q, n_cl2, fs, x, e, solver)


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


def run(instance, solver, start_bound=1, timeout=0, ub=maxsize, opt_size=False):
    c_bound = start_bound
    c_lb = 1
    best_model = None
    best_depth = None

    while c_lb < ub:
        print(f"Running depth {c_bound}")
        stdout.flush()
        with solver() as slv:
            vs = encode(instance, c_bound, slv)

            if timeout == 0:
                solved = slv.solve()
            else:
                def interrupt(s):
                    s.interrupt()

                timer = Timer(timeout, interrupt, [slv])
                timer.start()
                solved = slv.solve_limited(expect_interrupt=True)
                timer.cancel()
            if solved:
                model = {abs(x): x > 0 for x in slv.get_model()}
                best_model = _decode(model, instance, c_bound, vs)
                best_depth = c_bound
                ub = c_bound
                c_bound -= 1
            else:
                c_bound += 1
                c_lb = c_bound

    if opt_size and best_model:
        with solver() as slv:
            c_size_bound = best_model.root.get_leafs() - 1
            solved = True
            vs = encode(instance, best_depth, slv)
            card = encode_size(vs, instance, slv, best_depth)

            tot = ITotalizer(card, c_size_bound, top_id=vs["pool"].top+1)
            slv.append_formula(tot.cnf)

            while solved:
                print(f"Running size {c_size_bound}")
                stdout.flush()
                if timeout == 0:
                    solved = slv.solve()
                else:
                    def interrupt(s):
                        s.interrupt()

                    timer = Timer(timeout, interrupt, [slv])
                    timer.start()
                    solved = slv.solve_limited(expect_interrupt=True)
                    timer.cancel()

                if solved:
                    model = {abs(x): x > 0 for x in slv.get_model()}
                    best_model = _decode(model, instance, best_depth, vs)
                    c_size_bound -= 1
                    slv.add_clause([-tot.rhs[c_size_bound]])
                else:
                    break

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
        _alg1(instance, e_idx, c_bound, 0, 1, list(), vs["f"], vs["x"], vs["e"], slv)
        _alg2(instance, e_idx, c_bound, 0, 1, list(), vs["class_map"], vs["x"], vs["c"], slv)

    return guess


def run_limited(solver, strategy, size_limit, limit, start_bound=1, go_up=True, timeout=0):
    c_bound = start_bound
    best_model = None
    strategy.extend(limit[c_bound])

    while True:
        print(f"Running {c_bound}")
        with solver() as slv:
            while estimate_size(strategy.instance, c_bound) > size_limit and len(strategy.instance.examples) > 1:
                strategy.pop()

            vs = encode(strategy.instance, c_bound, slv)
            timed_out = []
            if timeout == 0:
                solved = slv.solve()
            else:
                def interrupt(s):
                    s.interrupt()
                    timed_out.append(True)

                timer = Timer(timeout, interrupt, [slv])
                timer.start()
                solved = slv.solve_limited(expect_interrupt=True)
                timer.cancel()

            if solved:
                model = {abs(x): x > 0 for x in slv.get_model()}
                best_model = _decode(model, strategy.instance, c_bound, vs)

                if go_up:
                    break

                strategy.extend(limit[c_bound-1] - limit[c_bound])
                c_bound -= 1
            else:
                if go_up:
                    if c_bound == len(limit) or timed_out:
                        for _ in range(0, 5):
                            strategy.pop()
                    else:
                        for _ in range(0, limit[c_bound] - limit[c_bound] + 1):
                            strategy.pop()
                        c_bound += 1
                else:
                    break

    return best_model


def _decode(model, instance, limit, vs):
    class_map = vs['class_map']
    fs = vs["f"]
    cs = vs["c"]
    es = vs["e"]

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
                        tree.set_root(real_f, tsh, real_f in instance.is_categorical or model[es[i]])
                    else:
                        tree.add_node(real_f, tsh, i // 2, i % 2 == 1, real_f in instance.is_categorical or model[es[i]])
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
                tree.add_leaf(c_c, (num_leafs + i)//2, i % 2 == 1)

    #_reduce_tree(tree, instance)
    tree.clean(instance)
    return tree


def _reduce_tree(tree, instance):
    assigned = {tree.root.id: list(instance.examples)}
    q = [tree.root]
    p = {tree.root.id: None}
    leafs = []

    while q:
        c_n = q.pop()
        examples = assigned[c_n.id]

        if not c_n.is_leaf:
            p[c_n.left.id] = c_n.id
            p[c_n.right.id] = c_n.id
            assigned[c_n.left.id] = []
            assigned[c_n.right.id] = []

            for e in examples:
                if e.features[c_n.feature]:
                    assigned[c_n.left.id].append(e)
                else:
                    assigned[c_n.right.id].append(e)

            q.append(c_n.right)
            q.append(c_n.left)
        else:
            leafs.append(c_n)

    for lf in leafs:
        # May already be deleted
        if tree.nodes[lf.id] is None:
            continue

        if len(assigned[lf.id]) == 0:
            c_p = tree.nodes[p[lf.id]]
            o_n = c_p.right if c_p.left.id == lf.id else c_p.left
            if p[c_p.id] is None:
                tree.root = o_n
                p[o_n.id] = None
            else:
                c_pp = tree.nodes[p[c_p.id]]
                if c_pp.left.id == c_p.id:
                    c_pp.left = o_n
                else:
                    c_pp.right = o_n
                p[o_n.id] = c_pp.id

            tree.nodes[lf.id] = None
            tree.nodes[c_p.id] = None


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
    f = instance.num_features

    forbidden_c = (2**lc - c) * d2 * lc
    alg1_lits = s * sum(2**i * f * (i+2) for i in range(0, depth))

    return d2 * f * (f-1) // 2 + d2 * f + forbidden_c + alg1_lits + s * d2 * (depth+1) * lc
