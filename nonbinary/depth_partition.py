from pysat.formula import IDPool

from nonbinary.decision_tree import DecisionTree


def _init_vars(instance, depth, vs, start=0):
    pool = IDPool() if not vs else vs["pool"]
    d = {} if not vs else vs["d"]

    for i in range(0, len(instance.examples)):
        d[i] = {}
        for dl in range(0, depth):
            d[i][dl] = {}
            for cf in range(1, instance.num_features + 1):
                if len(instance.domains[cf]) == 0:
                    continue

                # We don't need an entry for the last variable, as <= maxval is redundant
                for j in range(0, len(instance.domains[cf]) - (0 if cf in instance.is_categorical else 1)):
                    d[i][dl][instance.feature_idx[cf] + j] = pool.id(f"d{i}_{dl}_{instance.feature_idx[cf] + j}")

    if not vs:
        g = [{} for _ in range(0, len(instance.examples))]
    else:
        g = vs["g"]
        g.extend({} for _ in range(start, len(instance.examples)))

    for i in range(0, len(instance.examples)):
        for j in range(i + 1, len(instance.examples)):
            g[i][j] = [pool.id(f"g{i}_{j}_{d}") for d in range(0, depth + 1)]

    return g, d, pool


def encode(instance, depth, solver, opt_size, start=0, vs=None):
    g, d, p = _init_vars(instance, depth, vs, start)

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
                for cf in range(1, instance.num_features + 1):
                    if len(instance.domains[cf]) == 0:
                        continue

                    # We don't need an entry for the last variable, as <= maxval is redundant
                    for k in range(0, len(instance.domains[cf]) - (0 if cf in instance.is_categorical else 1)):
                        if cf in instance.is_categorical and \
                                (instance.examples[i].features[cf] == instance.domains[cf][k]) == (instance.examples[j].features[cf] == instance.domains[cf][k]):
                            solver.add_clause([-g[i][j][dl], -d[i][dl][instance.feature_idx[cf] + k], g[i][j][dl + 1]])
                        elif cf not in instance.is_categorical and \
                            (instance.examples[i].features[cf] <= instance.domains[cf][k]) == (instance.examples[j].features[cf] <= instance.domains[cf][k]):
                            solver.add_clause([-g[i][j][dl], -d[i][dl][instance.feature_idx[cf] + k], g[i][j][dl + 1]])
                        else:
                            solver.add_clause([-d[i][dl][instance.feature_idx[cf] + k], -g[i][j][dl + 1]])

    # Verify that group cannot merge
    for i in range(0, len(instance.examples)):
        for j in range(max(start, i + 1), len(instance.examples)):
            for dl in range(0, depth):
                solver.add_clause([g[i][j][dl], -g[i][j][dl + 1]])

    # Verify that d is consistent
    for i in range(0, len(instance.examples)):
        for j in range(max(start, i + 1), len(instance.examples)):
            for dl in range(0, depth):
                for f in d[i][dl].keys():
                    solver.add_clause([-g[i][j][dl], -d[i][dl][f], d[j][dl][f]])

    # One feature per level and group
    for i in range(start, len(instance.examples)):
        for dl in range(0, depth):
            clause = []
            for f1, f1val in d[i][dl].items():
                clause.append(f1val)
                # This set of clauses is not needed for correctness but is faster for small complex instances
                for f2, f2val in d[i][dl].items():
                    if f1 < f2:
                        solver.add_clause([-f1val, -f2val])
            solver.add_clause(clause)

    return {"g": g, "d": d, "pool": p}


def encode_size(vs, instance, solver, dl):
    pool = vs["pool"]
    card_vars = [pool.id("s0")]

    solver.add_clause([pool.id(f"s0")])
    for i in range(1, len(instance.examples)):
        clause = [vs["g"][j][i][dl] for j in range(0, i)]
        clause.append(pool.id(f"s{i}"))
        solver.add_clause(clause)
        card_vars.append(pool.id(f"s{i}"))

    return card_vars


def estimate_size_add(instance, dl):
    n = len(instance.examples)
    # Last part is for the totalizer
    return n * (n-1) // 2 + n + n * n * 3


def _decode(model, instance, depth, vs):
    tree = DecisionTree()
    g = vs["g"]
    ds = vs["d"]

    # Root
    def find_feature(ce, cdl):
        for cf, cfv in ds[ce][cdl].items():
            if model[cfv]:
                if instance.num_features == 1:
                    real_f = 1
                    tsh = instance.domains[1][cf - instance.feature_idx[1]]
                    return real_f, tsh
                else:
                    for r_f in range(2, instance.num_features+1):
                        real_f = None
                        if instance.feature_idx[r_f - 1] <= cf < instance.feature_idx[r_f]:
                            real_f = r_f - 1
                        elif r_f == instance.num_features:
                            real_f = r_f
                        if real_f is not None:
                            tsh = instance.domains[real_f][cf - instance.feature_idx[real_f]]
                            return real_f, tsh
                # else:
                #     print(f"ERROR double feature {cf} and {ce_feature} for experiment {ce}, at level {cdl}.")

        return None

    def df_tree(grp, parent, d):
        if d == depth:
            cls = grp[0][1].cls
            for _, e in grp:
                if e.cls != cls:
                    print(f"Error, double cls in leaf group {cls}, {e.cls}")
                    exit(1)

            # This is the edge case, where all samples have the same class, we reached the leaf without splitting
            if parent is None:
                tree.set_root_leaf(cls)
            else:
                is_left = grp[0][1].features[parent.feature] == parent.threshold if parent.feature in instance.is_categorical else \
                    grp[0][1].features[parent.feature] <= parent.threshold
                tree.add_leaf(cls, parent.id, is_left)

            return

        # Find feature
        f, f_t = find_feature(grp[0][0], d)

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
            is_left = grp[0][1].features[
                          parent.feature] == parent.threshold if parent.feature in instance.is_categorical else \
                grp[0][1].features[parent.feature] <= parent.threshold
        else:
            is_left = None

        if parent is not None:
            for ng in new_grps:
                for _, e in ng:
                    if is_left:
                        if parent.feature in instance.is_categorical:
                            if e.features[parent.feature] != parent.threshold:
                                print(f"Inhomogenous group, values {e.features[f]}")
                                exit(1)
                        else:
                            if e.features[parent.feature] > parent.threshold:
                                print(f"Inhomogenous group, values {e.features[f]}")
                                exit(1)
                    else:
                        if parent.feature in instance.is_categorical:
                            if e.features[parent.feature] == parent.threshold:
                                print(f"Inhomogenous group, values {e.features[f]}")
                                exit(1)
                        else:
                            if e.features[parent.feature] <= parent.threshold:
                                print(f"Inhomogenous group, values {e.features[f]}")
                                exit(1)

        if len(new_grps) > 1:
            if parent is None:
                n_n = tree.set_root(f, f_t)
            else:
                n_n = tree.add_node(f, f_t, parent.id, is_left, f in instance.is_categorical)

            for ng in new_grps:
                df_tree(ng, n_n, d+1)
        else:
            df_tree(new_grps[0], parent, d+1)

    df_tree(list(enumerate(instance.examples)), None, 0)
    tree.clean(instance)
    return tree


def check_consistency(model, instance, num_nodes, tree):
    pass


def estimate_size(instance, depth, start=0):
    """Estimates the size in the number of literals the encoding will require."""
    f = sum(len(instance.domains[x]) for x in range(1, instance.num_features+1))
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


def lb():
    return 1


def increment():
    return 1


def is_sat():
    return True


def get_tree_size(tree):
    return tree.root.get_leaves()
