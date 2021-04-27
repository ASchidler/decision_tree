from decision_tree import DecisionTree
from pysat.formula import IDPool
from sys import maxsize
from threading import Timer


def _init_vars(instance, depth, pool, start=0):
    if not pool:
        pool = IDPool()
    d = {}
    for i in range(0, len(instance.examples)):
        d[i] = {}
        for dl in range(0, depth):
            d[i][dl] = {}
            for f in range(1, instance.num_features + 1):
                d[i][dl][f] = pool.id(f"d{i}_{dl}_{f}")

    g = [{} for _ in range(0, len(instance.examples))]
    for i in range(0, len(instance.examples)):
        for j in range(i + 1, len(instance.examples)):
            g[i][j] = [pool.id(f"g{i}_{j}_{d}") for d in range(0, depth + 1)]

    return g, d, pool


def encode(instance, depth, solver, start=0, pool=None):
    g, d, p = _init_vars(instance, depth, pool, start)

    # Add level 0, all examples are in the same group
    for i in range(0, len(instance.examples)):
        for j in range(max(start, i + 1), len(instance.examples)):
            solver.add_clause([g[i][j][0]])

    # Verify that at last level, the partitioning is by class
    for i in range(0, len(instance.examples)):
        for j in range(max(start, i + 1), len(instance.examples)):
            if instance.examples[i].cls != instance.examples[j].cls:
                solver.add_clause([-g[i][j][depth]])

    # Verify that the examples are partitioned correctly
    for i in range(0, len(instance.examples)):
        for j in range(max(start, i + 1), len(instance.examples)):
            for dl in range(0, depth):
                for f in range(1, instance.num_features+1):
                    if instance.examples[i].features[f] == instance.examples[j].features[f]:
                        solver.add_clause([-g[i][j][dl], -d[i][dl][f], g[i][j][dl+1]])
                    else:
                        solver.add_clause([-d[i][dl][f], -g[i][j][dl + 1]])

    # Verify that group cannot merge
    for i in range(0, len(instance.examples)):
        for j in range(max(start, i + 1), len(instance.examples)):
            for dl in range(0, depth):
                solver.add_clause([g[i][j][dl], -g[i][j][dl + 1]])

    # Verify that d is consistent
    for i in range(0, len(instance.examples)):
        for j in range(max(start, i + 1), len(instance.examples)):
            for dl in range(0, depth):
                for f in range(1, instance.num_features+1):
                    solver.add_clause([-g[i][j][dl], -d[i][dl][f], d[j][dl][f]])

    # One feature per level and group
    for i in range(start, len(instance.examples)):
        for dl in range(0, depth):
            clause = []
            for f in range(1, instance.num_features + 1):
                clause.append(d[i][dl][f])
                # This set of clauses is not needed for correctness but is faster for small complex instances
                for f2 in range(f + 1, instance.num_features + 1):
                    solver.add_clause([-d[i][dl][f], -d[i][dl][f2]])
            solver.add_clause(clause)

    return {"g": g, "d": d, "p": p}


def run(instance, solver, start_bound=1, timeout=0, ub=maxsize):
    c_bound = start_bound
    clb = 0
    best_model = None

    while clb < ub:
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
                clb = c_bound + 1
                c_bound += 1

    return best_model


def _decode(model, instance, depth, vs):
    tree = DecisionTree(instance.num_features, 1)
    g = vs["g"]
    ds = vs["d"]

    # Root
    def find_feature(ce, cdl):
        ce_feature = None
        for cf in range(1, instance.num_features+1):
            if model[ds[ce][cdl][cf]]:
                if ce_feature is None:
                    ce_feature = cf
                # else:
                #     print(f"ERROR double feature {cf} and {ce_feature} for experiment {ce}, at level {cdl}.")
        if ce_feature is None:
            print(f"ERROR no feature for {ce} at level {cdl}.")
        return ce_feature

    def df_tree(grp, parent, d):
        if d == depth:
            cls = grp[0][1].cls
            for _, e in grp:
                if e.cls != cls:
                    print(f"Error, double cls in leaf group {cls}, {e.cls}")

            # This is the edge case, where all samples have the same class, we reached the leaf without splitting
            if parent is None:
                p_f = find_feature(grp[0][0], 0)
                tree.set_root(p_f)
                parent = tree.nodes[1]

                o_val = not grp[0][1].features[parent.feature]
                tree.nodes.append(None)
                tree.add_leaf(len(tree.nodes) - 1, parent.id, o_val, cls)

            tree.nodes.append(None)
            val = grp[0][1].features[parent.feature]
            tree.add_leaf(len(tree.nodes) - 1, parent.id, val, cls)

            return

        # Find feature
        f = find_feature(grp[0][0], d)

        # Find groups
        new_grps = []

        for e_id, e in grp:
            found = False
            for ng in new_grps:
                n_id, _ = ng[0]
                u = min(e_id, n_id)
                v = max(e_id, n_id)

                if model[g[u][v][d+1]]:
                    if found:
                        print("Double group membership")
                        exit(1)
                    found = True
                    ng.append((e_id, e))
            if not found:
                new_grps.append([(e_id, e)])

        # Check group consistency
        if parent is not None:
            for ng in new_grps:
                val = ng[0][1].features[parent.feature]

                for _, e in ng:
                    if e.features[parent.feature] != val:
                        print(f"Inhomogenous group, values {val}, {e.features[f]}")
                        exit(1)

        if len(new_grps) > 1:
            if parent is None:
                tree.set_root(f)
                n_n = tree.nodes[1]
            else:
                val = grp[0][1].features[parent.feature]
                tree.nodes.append(None)
                n_n = tree.add_node(len(tree.nodes) - 1, parent.id, f, val)
            for ng in new_grps:
                df_tree(ng, n_n, d+1)
        else:
            df_tree(new_grps[0], parent, d+1)

    df_tree(list(enumerate(instance.examples)), None, 0)

    return tree


def check_consistency(model, instance, num_nodes, tree):
    pass


def estimate_size(instance, depth, start=0):
    """Estimates the size in the number of literals the encoding will require."""
    f = instance.num_features
    s = len(instance.examples) - start
    s2 = s * (s-1)//2

    return s2 + s2 + s2*depth*f*3 + s2*depth*2 + s2 * depth * f * 3 + s*depth*f + s * depth * f * (f-1) // 2


def new_bound(tree, instance):
    if tree is None:
        return 1

    def dfs_find(node, level):
        if node.is_leaf:
            return level
        else:
            return max(dfs_find(node.left, level + 1), dfs_find(node.right, level + 1))

    return dfs_find(tree.root, 0)


def max_instances(num_features, limit):
    if num_features < 20:
        return 50
    if num_features < 35:
        return 40
    return 25


def lb():
    return 1
