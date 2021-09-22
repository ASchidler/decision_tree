import math

from decision_tree import DecisionTree
import itertools
from pysat.formula import IDPool
from sys import maxsize
from threading import Timer


class _InternalData:
    def __init__(self):
        self.v = None
        self.left = None
        self.right = None
        self.p = None
        self.a = None
        self.u = None
        self.d0 = None
        self.d1 = None
        self.c = None
        self.increment = 2
        self.class_map = None
        self.pool = IDPool()


def _init_vars(instance, num_nodes, c_var):    
    data = _InternalData()
    # First the tree structure
    data.v = {}
    for i in range(1, num_nodes + 1):
        data.v[i] = data.pool.id(f"v{i}")

    data.right = {i: {} for i in range(1, num_nodes + 1)}
    data.left = {i: {} for i in range(1, num_nodes + 1)}

    for i in range(1, num_nodes + 1):
        for j in range(i+1, min(2 * i + 2, num_nodes + 1)):
            if j % 2 == 0:
                data.left[i][j] = data.pool.id(f"left{i}_{j}")
            else:
                data.right[i][j] = data.pool.id(f"right{i}_{j}")

    data.p = [{} for _ in range(0, num_nodes + 1)]
    for j in range(2, num_nodes + 1, 2): # Starts from 2 not 1
        for i in range(j // 2, j):
            data.p[j][i] = data.pool.id(f"p{j}_{i}")
            data.p[j+1][i] = data.pool.id(f"p{j+1}_{i}")

    # Now the decision tree
    data.a = {}
    data.u = {}
    data.d0 = {}
    data.d1 = {}

    for cf in range(1, instance.num_features + 1):
        if len(instance.domains[cf]) == 0:
            continue
        for k in range(0, len(instance.domains[cf]) - (0 if cf in instance.is_categorical else 1)):
            i = instance.feature_idx[cf] + k
            data.a[i] = {}
            data.u[i] = {}
            data.d0[i] = {}
            data.d1[i] = {}

            for j in range(1, num_nodes + 1):
                data.a[i][j] = data.pool.id(f"a{i}_{j}")
                data.u[i][j] = data.pool.id(f"u{i}_{j}")
                data.d0[i][j] = data.pool.id(f"d0_{i}_{j}")
                data.d1[i][j] = data.pool.id(f"d1_{i}_{j}")
            data.c = {i: [data.pool.id(f"c{i}_{j}") for j in range(0, c_var)] for i in range(1, num_nodes+1)}

    return data


def lr(i, mx):
    if i % 2 == 0:
        return range(i + 2, min(2 * i, mx - 1) + 1, 2)
    else:
        return range(i + 1, min(2 * i, mx - 1) + 1, 2)


def rr(i, mx):
    if i % 2 == 0:
        return range(i + 3, min(2 * i + 1, mx) + 1, 2)
    else:
        return range(i + 2, min(2 * i + 1, mx) + 1, 2)


def pr(j):
    if j % 2 == 0:
        return range(max(1, j // 2), j)
    else:
        return range(max(1, (j - 1) // 2), j-1)


def _encode_tree_structure(data, num_nodes, solver):
    # root is not a leaf
    solver.add_clause([-data.v[1]])

    for i in range(1, num_nodes + 1):
        for j in lr(i, num_nodes):
            # Leafs have no children
            solver.add_clause([-data.v[i], -data.left[i][j]])
            # children are consecutively numbered
            solver.add_clause([-data.left[i][j], data.right[i][j + 1]])
            solver.add_clause([data.left[i][j], -data.right[i][j + 1]])

    # Enforce parent child relationship
    for i in range(1, num_nodes + 1):
        for j in lr(i, num_nodes):
            solver.add_clause([-data.p[j][i], data.left[i][j]])
            solver.add_clause([data.p[j][i], -data.left[i][j]])
        for j in rr(i, num_nodes):
            solver.add_clause([-data.p[j][i], data.right[i][j]])
            solver.add_clause([data.p[j][i], -data.right[i][j]])

    # Cardinality constraint
    # Each non leaf must have exactly one left child
    for i in range(1, num_nodes + 1):
        # First must have a child
        nodes = []
        for j in lr(i, num_nodes):
            nodes.append(data.left[i][j])
        solver.add_clause([data.v[i], *nodes])
        # Next, not more than one
        for j1 in lr(i, num_nodes):
            for j2 in lr(i, num_nodes):
                if j2 > j1:
                    solver.add_clause([data.v[i], -data.left[i][j1], -data.left[i][j2]])

    # Each non-root must have exactly one parent
    for j in range(2, num_nodes + 1, 2):
        clause1 = []
        clause2 = []
        for i in range(j//2, j):
            clause1.append(data.p[j][i])
            clause2.append(data.p[j+1][i])
        solver.add_clause([*clause1])
        solver.add_clause([*clause2])

        for i1 in range(j//2, j):
            for i2 in range(i1 + 1, j):
                solver.add_clause([-data.p[j][i1], -data.p[j][i2]])
                solver.add_clause([-data.p[j+1][i1], -data.p[j+1][i2]])


def _encode_discriminating(data, instance, num_nodes, solver):
    for r in data.d0.keys():
        solver.add_clause([-data.d0[r][1]])
        solver.add_clause([-data.d1[r][1]])

    # Discriminating features
    for j in range(2, num_nodes + 1, 2):
        for r in data.d1.keys():
            for direction in [False, True]:
                jpathl = data.d1[r][j] if direction else data.d0[r][j]
                jpathr = data.d1[r][j+1] if direction else data.d0[r][j+1]
                for i in pr(j):
                    ipath = data.d1[r][i] if direction else data.d0[r][i]

                    # Children inherit from the parent
                    solver.add_clause([-data.left[i][j], -ipath, jpathl])
                    solver.add_clause([-data.right[i][j+1], -ipath, jpathr])

                    if direction:
                        # The current node discriminates
                        solver.add_clause([-data.left[i][j], -data.a[r][i], jpathl])
                        # Other side of the implication
                        solver.add_clause([-jpathl, -data.left[i][j], ipath, data.a[r][i]])
                        solver.add_clause([-jpathr, -data.right[i][j+1], ipath])
                    else:
                        solver.add_clause([-data.right[i][j+1], -data.a[r][i], jpathr])
                        # Other side of the implication
                        solver.add_clause([-jpathl, -data.left[i][j], ipath])
                        solver.add_clause([-jpathr, -data.right[i][j + 1], ipath, data.a[r][i]])


def _encode_feature(data, instance, num_nodes, solver):
    # Feature assignment
    # u means that the feature already occurred in the current branch
    for r in data.a.keys():
        for j in range(1, num_nodes + 1):
            # Using the feature sets u in rest of sub-tree
            solver.add_clause([-data.a[r][j], data.u[r][j]])

            for i in pr(j):
                # If u is true for the parent, the feature must not be used by any child
                solver.add_clause([-data.u[r][i], -data.p[j][i], -data.a[r][j]])

                # Inheritance of u from parent to child
                solver.add_clause([-data.u[r][i], -data.p[j][i], data.u[r][j]])
                # Other side of the equivalence, if urj is true, than one of the conditions must hold
                solver.add_clause([-data.u[r][j], -data.p[j][i], data.a[r][j], data.u[r][i]])


    # Leafs have no feature
    for r in data.a.keys():
        for j in range(1, num_nodes + 1):
            solver.add_clause([-data.v[j], -data.a[r][j]])

    # Non-Leafs have exactly one feature
    for j in range(1, num_nodes + 1):
        clause = [data.v[j]]
        for r in data.a.keys():
            clause.append(data.a[r][j])
            for r2 in data.a.keys():
                if r2 > r:
                    solver.add_clause([-data.a[r][j], -data.a[r2][j]])
        solver.add_clause([*clause])


def _encode_examples(data, instance, num_nodes, solver, start=0):
    for ie in range(start, len(instance.examples)):
        e = instance.examples[ie]

        for j in range(1, num_nodes + 1):
            # If the class of the leaf differs from the class of the example, at least one
            # node on the way must discriminate against the example, otherwise the example
            # could be classified wrong
            clause = [-data.v[j]]

            for cf in range(1, instance.num_features + 1):
                if len(instance.domains[cf]) == 0:
                    continue

                # We don't need an entry for the last variable, as <= maxval is redundant
                for k in range(0, len(instance.domains[cf]) - (0 if cf in instance.is_categorical else 1)):
                    r = instance.feature_idx[cf] + k

                    if cf in instance.is_categorical:
                        clause.append(data.d0[r][j] if e.features[cf] == instance.domains[cf][k] else data.d1[r][j])
                    else:
                        clause.append(data.d0[r][j] if e.features[cf] <= instance.domains[cf][k] else data.d1[r][j])

            ec = data.class_map[e.cls]
            for c in range(0, len(ec)):
                solver.add_clause([*clause, data.c[j][c] if ec[c] else -data.c[j][c]])


def _improve(data, num_nodes, solver):
    ld = [None]
    for i in range(1, num_nodes + 1):
        ld.append([])
        for t in range(0, i//2 + 1):
            ld[i].append(data.pool.id(f"ld{i}_{t}"))
            if t == 0:
                solver.add_clause([ld[i][t]])
            else:
                if i > 1:
                    solver.add_clause([-ld[i - 1][t - 1], -data.v[i], ld[i][t]])
                    if t < len(ld[i-1]):
                        # Carry over
                        solver.add_clause([-ld[i-1][t], ld[i][t]])
                        # Increment if leaf
                        # i == 1 cannot be a leaf, as it is the root
                        solver.add_clause([-ld[i][t], ld[i-1][t], ld[i-1][t-1]])
                        solver.add_clause([-ld[i][t], ld[i-1][t], data.v[i]])
                    else:
                        solver.add_clause([-ld[i][t], ld[i - 1][t - 1]])
                        solver.add_clause([-ld[i][t], data.v[i]])
                # Use bound
                if 2*(i-t+1) <= num_nodes:
                    solver.add_clause([-ld[i][t], -data.left[i][2*(i-t+1)]])
                if 2*(i-t+1)+1 <= num_nodes:
                    solver.add_clause([-ld[i][t], -data.right[i][2*(i-t+1)+1]])

    tau = [None]
    for i in range(1, num_nodes+1):
        tau.append([])
        for t in range(0, i+1):
            tau[i].append(data.pool.id(f"tau{i}_{t}"))
            if t == 0:
                solver.add_clause([tau[i][t]])
            else:
                if i > 1:
                    # Increment
                    solver.add_clause([-tau[i - 1][t - 1], data.v[i], -tau[i][t]])
                    if t < len(tau[i-1]):
                        # Carry over
                        solver.add_clause([-tau[i-1][t], tau[i][t]])

                        # Reverse equivalence
                        solver.add_clause([-tau[i][t], tau[i-1][t], tau[i-1][t-1]])
                        solver.add_clause([-tau[i][t], tau[i-1][t], -data.v[i]])
                    else:
                        # Reverse equivalence
                        solver.add_clause([-tau[i][t], tau[i - 1][t - 1]])
                        solver.add_clause([-tau[i][t], -data.v[i]])

            if t > (i//2) + (i % 2): # i/2 rounded up
                # Use bound
                if num_nodes >= 2*(t - 1) > i:
                    solver.add_clause([-tau[i][t], -data.left[i][2*(t-1)]])
                if i < 2*t-1 <= num_nodes:
                    solver.add_clause([-tau[i][t], -data.right[i][2*t-1]])

    # root is the first non-leaf
    #solver.add_clause([tau[1][1]])


def encode(instance, num_nodes, solver, opt_size, improve=True):
    classes = set()
    for e in instance.examples:
        classes.add(e.cls)


    c_vars = len(bin(len(classes) - 1)) - 2  # "easier" than log_2
    classes = list(classes)  # Give classes an order
    classes.sort()
    data = _init_vars(instance, num_nodes, c_vars)

    data.class_map = {}

    for i in range(0, len(classes)):
        data.class_map[classes[i]] = []
        for c_v in bin(i)[2:][::-1]:
            if c_v == "1":
                data.class_map[classes[i]].append(True)
            else:
                data.class_map[classes[i]].append(False)

        while len(data.class_map[classes[i]]) < c_vars:
            data.class_map[classes[i]].append(False)

    _encode_tree_structure(data, num_nodes, solver)
    _encode_discriminating(data, instance, num_nodes, solver)
    _encode_feature(data, instance, num_nodes, solver)
    _encode_examples(data, instance, num_nodes, solver)
    if improve:
        _improve(data, num_nodes, solver)
    # TODO: Check why this causes UNSAT
    #_uniquify_classes(instance, num_nodes, data.class_map, solver)
    return data


def _uniquify_classes(data, instance, num_nodes, class_map, solver):
    # Disallow wrong labels
    # Forbid non-existing classes
    # Generate all class identifiers
    c_vars = len(next(iter(class_map.values())))
    for c_c in itertools.product([True, False], repeat=c_vars):
        # Check if identifier is used
        exists = False
        for c_v in data.class_map.values():
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
            for i in range(1, num_nodes + 1):
                clause = [-data.v[i]]
                for c in range(0, c_vars):
                    clause.append(-data.c[i][c] if c_c[c] else data.c[i][c])
                solver.add_clause([*clause])

    # Avoid for specific classes to be used only once
    for i in range(1, num_nodes + 1):
        for ck, cv in class_map.items():
            if ck < 0:
                continue
            for j in range(i + 1, num_nodes + 1):
                clause = [-data.v[i], -data.v[j]]
                for c in range(0, len(cv)):
                    modifier = -1 if cv[c] else 1

                    clause.append(modifier * data.c[i][c])
                    clause.append(modifier * data.c[j][c])
                solver.add_clause([*clause])


def extend(slv, instance, vs, c_bound, increment, size_limit):
    f = instance.num_features
    guess = increment * c_bound * (f + 1) * len(instance.classes)
    if guess > size_limit:
        return None

    _encode_examples(vs, instance, c_bound, slv, len(instance.examples)-increment)
    return guess


def encode_size(vs, instance, solver, dl):
    solver.add_clause([-1])
    solver.add_clause([1])


def _decode(model, instance, num_nodes, data):
    # TODO: This could be faster, but for debugging purposes, check for consistency
    tree = DecisionTree()
    def to_f(x):
        if instance.num_features == 1:
            real_f = 1
            tsh = instance.domains[1][r - instance.feature_idx[1]]
            return real_f, tsh
        else:
            for r_f in range(2, instance.num_features + 1):
                real_f = None
                if instance.feature_idx[r_f - 1] <= r < instance.feature_idx[r_f]:
                    real_f = r_f - 1
                elif r_f == instance.num_features:
                    real_f = r_f
                if real_f is not None:
                    tsh = instance.domains[real_f][r - instance.feature_idx[real_f]]
                    return real_f, tsh
        return None

    # Set root
    for r in data.a.keys():
        if model[data.a[r][1]]:
            real_f, tsh = to_f(r)

            if tree.root is None:
                tree.set_root(real_f, tsh)
            else:
                print(f"ERROR: Duplicate feature for root, set feature {tree.root.feature}, current {r}")

    if tree.root is None:
        print(f"ERROR: No feature found for root")

    # Add other nodes
    for j in range(2, num_nodes + 1):
        is_leaf = model[data.v[j]]

        parent = None
        for i in pr(j):
            if model[data.p[j][i]]:
                if parent is None:
                    parent = i
                else:
                    print(f"ERROR: Double parent for {j}, set {parent} also found {i}")
        if parent is None:
            print(f"ERROR: No parent found for {j}")
            raise

        feature = None
        if (j % 2 == 0 and not model[data.left[parent][j]]) or (j % 2 == 1 and not model[data.right[parent][j]]):
            print(f"ERROR: Parent - Child relationship mismatch, parent {parent}, child {j}")

        if not is_leaf:
            for r in data.a.keys():
                if model[data.a[r][j]]:
                    if feature is None:
                        real_f, tsh = to_f(r)
                        tree.add_node(real_f, tsh, parent, j % 2 == 0, real_f in instance.is_categorical)
                        feature = r
                    else:
                        print(f"ERROR: Duplicate feature for {j}, set feature {feature}, current {r}")
        else:
            c_c = None
            for k, v in data.class_map.items():
                failed = False
                for c in range(0, len(v)):
                    if model[data.c[j][c]] != v[c]:
                        failed = True
                if not failed:
                    c_c = k
            assert c_c is not None
            tree.add_leaf(c_c, parent, j % 2 == 0)

    _check_consistency(data, model, num_nodes, tree, instance)
    return tree


def _check_consistency(data, model, num_nodes, tree, instance):
    # Check left, right vars
    for j in range(2, num_nodes + 1):
        cnt = 0
        for i in pr(j):
            if j % 2 == 0 and model[data.left[i][j]]:
                cnt += 1
            elif j % 2 == 1 and model[data.right[i][j]]:
                cnt += 1

        if cnt != 1:
            print(f"Found non 1 child assignment of node {j}")

    # Check feature paths
    prev = [-1 for _ in range(0, num_nodes + 1)]
    for node in range(1, num_nodes + 1):
        if not tree.nodes[node].is_leaf:
            prev[tree.nodes[node].left.id] = node
            prev[tree.nodes[node].right.id] = node

    for node in range(2, num_nodes + 1):
        if tree.nodes[node].is_leaf:
            features = []
            thresholds = []
            values = []
            path = [node]
            cp = node
            # trace path from leaf to root
            while cp != 1:
                path.append(prev[cp])
                features.append(tree.nodes[prev[cp]].feature)
                thresholds.append(tree.nodes[prev[cp]].threshold)
                values.append(tree.nodes[prev[cp]].left.id == cp)
                cp = prev[cp]
            path.pop()
            path.reverse()
            features.reverse()
            thresholds.reverse()
            values.reverse()

            # Now verify the model
            for i in range(0, len(path)):
                feat = features[i]
                tsh = thresholds[i]
                cnode = path[i]
                d1val = values[i]
                d0val = not values[i]
                feat_id = instance.feature_idx[feat] + instance.domains[feat].index(tsh)

                for j in range(i, len(path)):
                    if not tree.nodes[path[j]].is_leaf and tree.nodes[path[j]].feature == feat and tree.nodes[path[j]].threshold == tsh:
                        print(f"ERROR duplicate feature {feat} in nodes {cnode} and {path[j]}")
                    if not model[data.u[feat_id][path[j]]]:
                        print(f"ERROR u for feature {feat} not set for node {path[j]}")
                    if model[data.d0[feat_id][path[j]]] != d0val:
                        print(f"ERROR d0 value wrong, feature {feat} at node {path[j]} is leaf: {tree.nodes[path[j]].is_leaf}")
                    if model[data.d1[feat_id][path[j]]] != d1val:
                        print(f"ERROR d1 value wrong, feature {feat} at node {path[j]} is leaf: {tree.nodes[path[j]].is_leaf}")


def new_bound(tree, instance):
    if tree is None:
        return 3

    return min(len(tree.nodes) - 1, 2 * 2**instance.num_features - 1)


def estimate_size_add(instance, dl):
    return 0


def lb():
    return 3


def increment():
    return 2


def max_instances(num_features, limit):
    if num_features < 50:
        return 100
    if num_features < 100:
        return 70
    return 50


def estimate_size(instance, size):
    """Estimates the required size in the number of literals"""
    guess = 0
    f = instance.num_features
    e = len(instance.examples)

    sum1 = sum(size - math.floor(math.log2(x)) for x in range(1, size + 1))
    guess += sum1 * 12
    guess += sum1 * sum1 * (sum1-1) // 2 * 3
    guess += sum(2 * (x - x//2) for x in range(1, size+1))
    guess += sum((x - x//2) * (x - x//2 - 1) // 2 * 4 for x in range(1, size+1))
    guess += 2 * f
    guess += f * sum(x//2 * 2 * 15 for x in range(1, size+1))
    guess += f * sum(x//2 for x in range(1, size+1)) * 10
    guess += 4 * f * size
    guess += f * (size + 1)
    guess += size * f * (f-1)
    guess += e * size * (f+1) * len(instance.classes)
    guess += size
    guess += sum(x//2 * ((x-1)//2 * 7 + 3 + (x - (x-1)//2) * 5 + 2) for x in range(1, size+1))
    guess += sum(x * ((x - 1) // 2 * 7 + 3 + (x - (x - 1) // 2) * 5 + 2) for x in range(1, size + 1))

    return guess

def is_sat():
    return True
