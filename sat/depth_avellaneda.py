import itertools
from sys import maxsize
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
        for j in range(1, instance.num_features + 1):
            f[i][j] = pool.id(f"f{i}_{j}")

    c_vars = len(next(iter(class_map.values())))
    c = {}
    for i in range(0, 2**limit):
        c[i] = {}
        for j in range(0, c_vars):
            c[i][j] = pool.id(f"c{i}_{j}")

    return x, f, c, pool


def encode(instance, limit, solver, opt_size=False):
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
        for f1 in range(1, instance.num_features + 1):
            clause.append(f[i][f1])
            for f2 in range(f1+1, instance.num_features + 1):
                solver.add_clause([-f[i][f1], -f[i][f2]])
        solver.add_clause(clause)

    for i in range(0, len(instance.examples)):
        _alg1(instance, i, limit, 0, 1, list(), f, x, solver)
        _alg2(instance, i, limit, 0, 1, list(), class_map, x, c, solver)

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
        if not example.features[f]:
            solver.add_clause([*clause, -x[e_idx][lvl], -fs[q][f]])
    n_cl = list(clause)
    n_cl.append(-x[e_idx][lvl])
    _alg1(instance, e_idx, limit, lvl+1, 2 * q + 1, n_cl, fs, x, solver)

    for f in range(1, instance.num_features + 1):
        if example.features[f]:
            solver.add_clause([*clause, x[e_idx][lvl], -fs[q][f]])
    n_cl2 = list(clause)
    n_cl2.append(x[e_idx][lvl])
    _alg1(instance, e_idx, limit, lvl+1, 2*q, n_cl2, fs, x, solver)


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
        print(f"Running {c_bound}")
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
                print(f"Running {c_size_bound}")
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
        _alg1(instance, e_idx, c_bound, 0, 1, list(), vs["f"], vs["x"], slv)
        _alg2(instance, e_idx, c_bound, 0, 1, list(), vs["class_map"], vs["x"], vs["c"], slv)

    return guess

#
# def run_incremental(instance, solver, strategy, timeout, size_limit, start_bound=1, increment=5, ubound=maxsize, opt_size=True):
#     c_bound = start_bound
#     best_model = (0.0, None)
#     c_solver = None
#     is_done = []
#     c = len(instance.classes)
#     lc = len(bin(c - 1)) - 2  # ln(c)
#     f = instance.num_features
#
#     def interrupt(set_done=True):
#         if c_solver is not None:
#             c_solver.interrupt()
#             if set_done:
#                 is_done.append(True)
#
#     def mini_interrupt():
#         # Increments the current bound
#         interrupt(set_done=False)
#
#     timer = Timer(timeout, interrupt)
#     timer.start()
#     new_best_model = None
#
#     while not is_done and c_bound <= ubound:
#         last_runtime = max(5, timeout / 5)
#         print(f"Running {c_bound}")
#         c_a = 0
#         with solver() as slv:
#             c_solver = slv
#
#             while estimate_size(strategy.instance, c_bound) > size_limit and len(strategy.instance.examples) > 1:
#                 strategy.instance.examples.pop()
#
#             c_guess = estimate_size(strategy.instance, c_bound)
#             d2 = 2 ** c_bound
#
#             solved = True
#             try:
#                 vs = encode(strategy.instance, c_bound, slv)
#             except MemoryError:
#                 solved = False
#                 c_bound -= 1
#                 for _ in range(0, len(strategy.instance.examples) // 5):
#                     strategy.instance.pop()
#
#             while solved:
#                 timer2 = Timer(5 * last_runtime, mini_interrupt)
#                 timer2.start()
#                 c_runtime_start = time.time()
#                 try:
#                     solved = slv.solve_limited(expect_interrupt=True)
#                 except MemoryError:
#                     solved = False
#                     c_bound -= 1  # Will be incremented at the end
#                     # Reduce instance size by 20%
#                     for _ in range(0, len(strategy.instance.examples) // 5):
#                         strategy.instance.pop()
#                 last_runtime = max(1.0, time.time() - c_runtime_start)
#
#                 timer2.cancel()
#                 if solved:
#                     model = {abs(x): x > 0 for x in slv.get_model()}
#                     new_best_model = _decode(model, strategy.instance, c_bound, vs)
#                     c_a = new_best_model.get_accuracy(instance.examples)
#                     if best_model[1] is None or best_model[0] < c_a:
#                         best_model = (c_a, new_best_model)
#
#                     # print(f"Found: a: {c_a}, d: {new_best_model.get_depth()}, n: {new_best_model.get_nodes()}")
#                     if c_a > 0.9999:
#                         break
#
#                     alg1_lits = increment * sum(2 ** i * f * (i + 2) for i in range(0, c_bound))
#                     c_guess += alg1_lits + increment * d2 * (c_bound + 1) * lc
#
#                     if c_guess > size_limit:
#                         is_done.append(True)
#                         break
#
#                     strategy.extend(increment, best_model[1])
#                     try:
#                         for e_idx in range(len(strategy.instance.examples)-increment, len(strategy.instance.examples)):
#                             vs["x"][e_idx] = {}
#                             for x2 in range(0, c_bound):
#                                 vs["x"][e_idx][x2] = vs["pool"].id(f"x{e_idx}{x2}")
#                             _alg1(strategy.instance, e_idx, c_bound, 0, 1, list(), vs["f"], vs["x"], slv)
#                             _alg2(strategy.instance, e_idx, c_bound, 0, 1, list(), vs["class_map"], vs["x"], vs["c"], slv)
#                     except MemoryError:
#                         is_done = True
#                         break
#                     # print(f"Extended to {len(strategy.instance.examples)}")
#
#             if c_a > 0.999999:
#                 is_done.append(True)
#
#             if opt_size and new_best_model is not None and is_done:
#                 timer.cancel()
#                 slv.clear_interrupt()
#                 c_size_bound = new_best_model.root.get_leafs() - 1
#                 solved = True
#                 card = encode_size(vs, strategy.instance, slv, c_bound)
#
#                 tot = ITotalizer(card, c_size_bound, top_id=vs["pool"].top + 1)
#                 slv.append_formula(tot.cnf)
#
#                 timer = Timer(timeout, interrupt)
#                 timer.start()
#
#                 while solved:
#                     print(f"Running {c_size_bound}")
#                     solved = slv.solve_limited(expect_interrupt=True)
#
#                     if solved:
#                         model = {abs(x): x > 0 for x in slv.get_model()}
#                         new_best_model = _decode(model, strategy.instance, c_bound, vs)
#                         c_a = new_best_model.get_accuracy(instance.examples)
#                         if best_model[1] is None or best_model[0] < c_a:
#                             best_model = (c_a, new_best_model)
#
#                         c_size_bound -= 1
#                         slv.add_clause([-tot.rhs[c_size_bound]])
#                     else:
#                         break
#             c_bound += 1
#     timer.cancel()
#     return best_model[1]


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

    num_leafs = 2**limit
    tree = DecisionTree(instance.num_features, 2 * num_leafs - 1)
    # Find features
    for i in range(1, num_leafs):
        f_found = False
        for f in range(1, instance.num_features+1):
            if model[fs[i][f]]:
                if f_found:
                    print(f"ERROR: double features found for node {i}, features {f} and {tree.nodes[i].feature}")
                else:
                    if i == 1:
                        tree.set_root(f)
                    else:
                        tree.add_node(i, i//2, f, i % 2 == 1)

    for c_c, c_v in class_map.items():
        for i in range(0, num_leafs):
            all_right = True
            for i_v in range(0, len(c_v)):
                if model[cs[i][i_v]] != c_v[i_v]:
                    all_right = False
                    break
            if all_right:
                tree.add_leaf(num_leafs + i, (num_leafs + i)//2, i % 2 == 1, c_c)

    _reduce_tree(tree, instance)

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
