import psutil
from pysat.formula import IDPool

from nonbinary import limits
from nonbinary.decision_tree import DecisionTree


def _init_vars(instance, depth, vs, start=0):
    pool = IDPool() if not vs else vs["pool"]
    d = {} if not vs else vs["d"]
    lx = {}

    for i in range(0, len(instance.examples)):
        d[i] = {}
        for dl in range(0, depth):
            if dl not in lx:
                lx[dl] = {}

            lx[dl][i] = pool.id(f"lx{dl}{i}")

            if psutil.Process().memory_info().vms > limits.mem_limit:
                return

            d[i][dl] = {}
            for cf in range(1, instance.num_features + 1):
                if len(instance.domains[cf]) == 0:
                    continue

                d[i][dl][cf] = pool.id(f"d{i}_{dl}_{cf}")

    if not vs:
        g = [{} for _ in range(0, len(instance.examples))]
    else:
        g = vs["g"]
        g.extend({} for _ in range(start, len(instance.examples)))

    for i in range(0, len(instance.examples)):
        if psutil.Process().memory_info().vms > limits.mem_limit:
            return
        for j in range(i + 1, len(instance.examples)):
            g[i][j] = [pool.id(f"g{i}_{j}_{d}") for d in range(0, depth + 1)]

    return g, d, pool, lx


def encode(instance, depth, solver, opt_size, start=0, vs=None):
    g, d, p, lx = _init_vars(instance, depth, vs, start)

    if psutil.Process().memory_info().vms > limits.mem_limit:
        return

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
    for cf in range(1, instance.num_features + 1):
        if len(instance.domains[cf]) == 0:
            continue

        if cf in instance.is_categorical:
            if psutil.Process().memory_info().vms > limits.mem_limit:
                return
            for dl in range(0, depth):
                for i in range(0, len(instance.examples)):
                    if instance.examples[i].features[cf] == "DummyValue":
                        solver.add_clause([-d[i][dl][cf], -lx[dl][i]])
                    else:
                        for j in range(max(start, i + 1), len(instance.examples)):
                            if instance.examples[i].features[cf] == instance.examples[j].features[cf]:
                                solver.add_clause([-g[i][j][dl], -d[i][dl][cf], -lx[dl][i], lx[dl][j]])
                                solver.add_clause([-g[i][j][dl], -d[i][dl][cf], -lx[dl][j], lx[dl][i]])
                            else:
                                solver.add_clause([-g[i][j][dl], -d[i][dl][cf], -lx[dl][i], -lx[dl][j]])
        else:
            for i in range(0, len(instance.examples)):
                assert(i == instance.examples[i].id)

            for dl in range(0, depth):
                if psutil.Process().memory_info().vms > limits.mem_limit:
                    return
                sorted_examples = sorted(instance.examples, key=lambda x: x.features[cf])
                for i in range(0, len(instance.examples)):
                    e1 = sorted_examples[i]
                    if e1.ignore:
                        continue

                    for j in range(max(start, i + 1), len(instance.examples)):
                        e2 = sorted_examples[j]
                        if e2.ignore:
                            continue

                        solver.add_clause([-g[min(e1.id, e2.id)][max(e1.id, e2.id)][dl], -d[e1.id][dl][cf], -lx[dl][e2.id], lx[dl][e1.id]])
                        #solver.add_clause([-g[min(e1.id, e2.id)][max(e1.id, e2.id)][dl], -d[e1.id][dl][cf], lx[dl][e1.id], -lx[dl][e2.id]])

                        if e1.features[cf] == e2.features[cf]:
                            solver.add_clause([-g[min(e1.id, e2.id)][max(e1.id, e2.id)][dl], -d[e1.id][dl][cf], -lx[dl][e1.id], lx[dl][e2.id]])
                            #solver.add_clause([-g[min(e1.id, e2.id)][max(e1.id, e2.id)][dl], -d[e1.id][dl][cf], lx[dl][e2.id], -lx[dl][e1.id]])

    # Verify that group cannot merge
    for i in range(0, len(instance.examples)):
        for j in range(max(start, i + 1), len(instance.examples)):
            for dl in range(0, depth):
                solver.add_clause([g[i][j][dl], -g[i][j][dl + 1]])
                solver.add_clause([-g[i][j][dl], -lx[dl][i], -lx[dl][j], g[i][j][dl + 1]])
                solver.add_clause([-g[i][j][dl], lx[dl][i], lx[dl][j], g[i][j][dl + 1]])
                solver.add_clause([-g[i][j][dl], lx[dl][i], -lx[dl][j], -g[i][j][dl + 1]])
                solver.add_clause([-g[i][j][dl], -lx[dl][i], lx[dl][j], -g[i][j][dl + 1]])

    # Verify that d is consistent
    if psutil.Process().memory_info().vms > limits.mem_limit:
        return

    for i in range(0, len(instance.examples)):
        for j in range(max(start, i + 1), len(instance.examples)):
            for dl in range(0, depth):
                for f in range(1, instance.num_features + 1):
                    solver.add_clause([-g[i][j][dl], -d[i][dl][f], d[j][dl][f]])

    if psutil.Process().memory_info().vms > limits.mem_limit:
        return

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

    return {"g": g, "d": d, "lx": lx, "pool": p}


def encode_size(vs, instance, solver, dl):
    if psutil.Process().memory_info().vms > limits.mem_limit:
        return

    pool = vs["pool"]
    card_vars = [pool.id("s0")]

    solver.add_clause([pool.id(f"s0")])
    for i in range(1, len(instance.examples)):
        clause = [vs["g"][j][i][dl] for j in range(0, i) if instance.examples[i].cls == instance.examples[j].cls]
        clause.append(pool.id(f"s{i}"))
        solver.add_clause(clause)
        card_vars.append(pool.id(f"s{i}"))

    return card_vars


def encode_extended_leaf_size(vs, instance, solver, dl):
    if psutil.Process().memory_info().vms > limits.mem_limit:
        return

    pool = vs["pool"]
    card_vars = []

    def is_extended_cls(ce):
        cls = instance.examples[ce].cls
        return cls.startswith("-") and cls != "EmptyLeaf"

    for i in range(1, len(instance.examples)):
        if is_extended_cls(i):
            clause = [vs["g"][j][i][dl] for j in range(0, i) if instance.examples[j].cls == instance.examples[i].cls]
            clause.append(pool.id(f"e{i}"))
            solver.add_clause(clause)
            card_vars.append(pool.id(f"e{i}"))

    return card_vars


def encode_extended_leaf_limit(vs, instance, solver, dl):
    if psutil.Process().memory_info().vms > limits.mem_limit:
        return

    def is_extended_cls(ce):
        cls = instance.examples[ce].cls
        return cls.startswith("-") and cls != "EmptyLeaf"

    for i in range(1, len(instance.examples)):
        if not is_extended_cls(i):
            continue

        for j in range(0, i):
            if instance.examples[j].cls == instance.examples[i].cls:
                solver.add_clause([vs["g"][j][i][dl]])


def estimate_size_add(instance, dl):
    n = len(instance.examples)
    # Last part is for the totalizer
    return n * (n-1) // 2 + n + n * n * 3


def _decode(model, instance, depth, vs):
    tree = DecisionTree()
    g = vs["g"]
    ds = vs["d"]
    lx = vs["lx"]

    # Root
    def find_feature(cg, cdl):
        ce = cg[0][0]
        for cf, cfv in ds[ce][cdl].items():
            if model[cfv]:
                if cf in instance.is_categorical:
                    for c_sample in cg:
                        if model[lx[cdl][c_sample[0]]]:
                            return cf, c_sample[1].features[cf]

                    return cf, "ThisWillBeReducedAway"
                else:
                    values = []
                    values2 = []
                    for c_sample in cg:
                        if model[lx[cdl][c_sample[0]]]:
                            values.append(c_sample[1].features[cf])
                        else:
                            values2.append(c_sample[1].features[cf])

                    tsh = max(values) if values else max(values2)
                    return cf, tsh

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
        f, f_t = find_feature(grp, d)

        # Find groups
        new_grps = []

        for e_id, e in grp:
            found = False
            for ng in new_grps:
                n_id, _ = ng[0]
                u = min(e_id, n_id)
                v = max(e_id, n_id)

                if model[g[u][v][d+1]]:
                    assert(model[lx[d][u]] == model[lx[d][u]])
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
                n_n = tree.set_root(f, f_t, f in instance.is_categorical)
            else:
                n_n = tree.add_node(f, f_t, parent.id, is_left, f in instance.is_categorical)

            for ng in new_grps:
                df_tree(ng, n_n, d+1)
        else:
            df_tree(new_grps[0], parent, d+1)

    first_group = [(i, e) for i, e in enumerate(instance.examples) if not e.ignore]
    df_tree(first_group, None, 0)
    # tree.clean(instance)
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
