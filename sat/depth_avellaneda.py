from decision_tree import DecisionTree
import itertools
from pysat.formula import IDPool, CNF
from sys import maxsize
from threading import Timer


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

    return x, f, c


def encode(instance, limit, solver):
    classes = list(instance.classes) # Give classes an order
    c_vars = len(bin(len(classes)-1)) - 2 # "easier" than log_2

    class_map = {}
    for i in range(0, len(classes)):
        class_map[classes[i]] = []
        for c_v in bin(i)[2:][::-1]:
            if c_v == "1":
                class_map[classes[i]].append(True)
            else:
                class_map[classes[i]].append(False)

        while len(class_map[classes[i]]) < c_vars:
            class_map[classes[i]].append(False)

    x, f, c = _init_var(instance, limit, class_map)

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
    # Generate all class identifiers
    for c_c in itertools.product([True, False], repeat=c_vars):
        # Check if identifier is used
        exists = False
        for c_v in class_map.values():
            all_match = True
            for i in range(0, c_vars):
                if c_v[i] != c_c[i]:
                    all_match = False
                    break
            if all_match:
                exists = True
                break
        # If identifier is not used, prevent it from being used
        if not exists:
            for c_n in range(0, 2**limit):
                clause = []
                for i in range(0, c_vars):
                    clause.append(c[c_n][i] if c_c[i] else -c[c_n][i])
                solver.add_clause(clause)
    return {"f": f, "x": x, "c": c, "class_map": class_map}


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


def run(instance, solver, start_bound=1, timeout=0, ub=maxsize):
    c_bound = start_bound
    c_lb = 0
    best_model = None

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
                ub = c_bound
                c_bound -= 1
            else:
                c_bound += 1
                c_lb = c_bound

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
