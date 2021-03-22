from decision_tree import DecisionTree
import itertools
from pysat.formula import IDPool, CNF
from sys import maxsize
from threading import Timer


class SizeNarodytska:
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
        self.formula = None

    def init_vars(self, instance, num_nodes, c_var):
        # First the tree structure
        self.v = {}
        for i in range(1, num_nodes + 1):
            self.v[i] = self.pool.id(f"v{i}")

        self.right = {i: {} for i in range(1, num_nodes + 1)}
        self.left = {i: {} for i in range(1, num_nodes + 1)}

        for i in range(1, num_nodes + 1):
            for j in range(i+1, min(2 * i + 2, num_nodes + 1)):
                if j % 2 == 0:
                    self.left[i][j] = self.pool.id(f"left{i}_{j}")
                else:
                    self.right[i][j] = self.pool.id(f"right{i}_{j}")

        self.p = [{} for _ in range(0, num_nodes + 1)]
        for j in range(2, num_nodes + 1, 2): # Starts from 2 not 1
            for i in range(j // 2, j):
                self.p[j][i] = self.pool.id(f"p{j}_{i}")
                self.p[j+1][i] = self.pool.id(f"p{j+1}_{i}")

        # Now the decision tree
        self.a = []
        self.u = []
        self.d0 = []
        self.d1 = []
        self.a.append(None)
        self.u.append(None)
        self.d0.append(None)
        self.d1.append(None)
        for i in range(1, instance.num_features + 1):
            self.a.append([])
            self.u.append([])
            self.d0.append([])
            self.d1.append([])
            self.a[i].append(None)
            self.u[i].append(None)
            self.d0[i].append(None)
            self.d1[i].append(None)
            for j in range(1, num_nodes + 1):
                self.a[i].append(self.pool.id(f"a{i}_{j}"))
                self.u[i].append(self.pool.id(f"u{i}_{j}"))
                self.d0[i].append(self.pool.id(f"d0_{i}_{j}"))
                self.d1[i].append(self.pool.id(f"d1_{i}_{j}"))
        self.c = {i: [self.pool.id(f"c{i}_{j}") for j in range(0, c_var)] for i in range(1, num_nodes+1)}

    @staticmethod
    def lr(i, mx):
        if i % 2 == 0:
            return range(i + 2, min(2 * i, mx - 1) + 1, 2)
        else:
            return range(i + 1, min(2 * i, mx - 1) + 1, 2)

    @staticmethod
    def rr(i, mx):
        if i % 2 == 0:
            return range(i + 3, min(2 * i + 1, mx) + 1, 2)
        else:
            return range(i + 2, min(2 * i + 1, mx) + 1, 2)

    @staticmethod
    def pr(j):
        if j % 2 == 0:
            return range(max(1, j // 2), j)
        else:
            return range(max(1, (j - 1) // 2), j-1)

    def encode_tree_structure(self, instance, num_nodes):
        # root is not a leaf
        self.formula.append([-self.v[1]])

        for i in range(1, num_nodes + 1):
            for j in SizeNarodytska.lr(i, num_nodes):
                # Leafs have no children
                self.formula.append([-self.v[i], -self.left[i][j]])
                # children are consecutively numbered
                self.formula.append([-self.left[i][j], self.right[i][j + 1]])
                self.formula.append([self.left[i][j], -self.right[i][j + 1]])

        # Enforce parent child relationship
        for i in range(1, num_nodes + 1):
            for j in SizeNarodytska.lr(i, num_nodes):
                self.formula.append([-self.p[j][i], self.left[i][j]])
                self.formula.append([self.p[j][i], -self.left[i][j]])
            for j in SizeNarodytska.rr(i, num_nodes):
                self.formula.append([-self.p[j][i], self.right[i][j]])
                self.formula.append([self.p[j][i], -self.right[i][j]])

        # Cardinality constraint
        # Each non leaf must have exactly one left child
        for i in range(1, num_nodes + 1):
            # First must have a child
            nodes = []
            for j in SizeNarodytska.lr(i, num_nodes):
                nodes.append(self.left[i][j])
            self.formula.append([self.v[i], *nodes])
            # Next, not more than one
            for j1 in SizeNarodytska.lr(i, num_nodes):
                for j2 in SizeNarodytska.lr(i, num_nodes):
                    if j2 > j1:
                        self.formula.append([self.v[i], -self.left[i][j1], -self.left[i][j2]])

        # Each non-root must have exactly one parent
        for j in range(2, num_nodes + 1, 2):
            clause1 = []
            clause2 = []
            for i in range(j//2, j):
                clause1.append(self.p[j][i])
                clause2.append(self.p[j+1][i])
            self.formula.append([*clause1])
            self.formula.append([*clause2])

            for i1 in range(j//2, j):
                for i2 in range(i1 + 1, j):
                    self.formula.append([-self.p[j][i1], -self.p[j][i2]])
                    self.formula.append([-self.p[j+1][i1], -self.p[j+1][i2]])

    def encode_discriminating(self, instance, num_nodes):
        for r in range(1, instance.num_features + 1):
            self.formula.append([-self.d0[r][1]])
            self.formula.append([-self.d1[r][1]])

        # Discriminating features
        for j in range(2, num_nodes + 1, 2):
            for r in range(1, instance.num_features + 1):
                for direction in [False, True]:
                    jpathl = self.d1[r][j] if direction else self.d0[r][j]
                    jpathr = self.d1[r][j+1] if direction else self.d0[r][j+1]
                    for i in SizeNarodytska.pr(j):
                        ipath = self.d1[r][i] if direction else self.d0[r][i]

                        # Children inherit from the parent
                        self.formula.append([-self.left[i][j], -ipath, jpathl])
                        self.formula.append([-self.right[i][j+1], -ipath, jpathr])

                        if direction:
                            # The current node discriminates
                            self.formula.append([-self.left[i][j], -self.a[r][i], jpathl])
                            # Other side of the implication
                            self.formula.append([-jpathl, -self.left[i][j], ipath, self.a[r][i]])
                            self.formula.append([-jpathr, -self.right[i][j+1], ipath])
                        else:
                            self.formula.append([-self.right[i][j+1], -self.a[r][i], jpathr])
                            # Other side of the implication
                            self.formula.append([-jpathl, -self.left[i][j], ipath])
                            self.formula.append([-jpathr, -self.right[i][j + 1], ipath, self.a[r][i]])

    def encode_feature(self, instance, num_nodes):
        # Feature assignment
        # u means that the feature already occurred in the current branch
        for r in range(1, instance.num_features + 1):
            for j in range(1, num_nodes + 1):
                # Using the feature sets u in rest of sub-tree
                self.formula.append([-self.a[r][j], self.u[r][j]])

                for i in SizeNarodytska.pr(j):
                    # If u is true for the parent, the feature must not be used by any child
                    self.formula.append([-self.u[r][i], -self.p[j][i], -self.a[r][j]])

                    # Inheritance of u from parent to child
                    self.formula.append([-self.u[r][i], -self.p[j][i], self.u[r][j]])
                    # Other side of the equivalence, if urj is true, than one of the conditions must hold
                    self.formula.append([-self.u[r][j], -self.p[j][i], self.a[r][j], self.u[r][i]])


        # Leafs have no feature
        for r in range(1, instance.num_features + 1):
            for j in range(1, num_nodes + 1):
                self.formula.append([-self.v[j], -self.a[r][j]])

        # Non-Leafs have exactly one feature
        for j in range(1, num_nodes + 1):
            clause = [self.v[j]]
            for r in range(1, instance.num_features + 1):
                clause.append(self.a[r][j])
                for r2 in range(r + 1, instance.num_features + 1):
                    self.formula.append([-self.a[r][j], -self.a[r2][j]])
            self.formula.append([*clause])

    def encode_examples(self, instance, num_nodes, class_map):
        for e in instance.examples:
            for j in range(1, num_nodes + 1):
                # If the class of the leaf differs from the class of the example, at least one
                # node on the way must discriminate against the example, otherwise the example
                # could be classified wrong
                clause = [-self.v[j]]

                for r in range(1, instance.num_features + 1):
                    clause.append(self.d0[r][j] if e.features[r] else self.d1[r][j])

                ec = class_map[e.cls]
                for c in range(0, len(ec)):
                    self.formula.append([*clause, self.c[j][c] if ec[c] else -self.c[j][c]])

    def improve(self, num_nodes):
        ld = [None]
        for i in range(1, num_nodes + 1):
            ld.append([])
            for t in range(0, i//2 + 1):
                ld[i].append(self.pool.id(f"ld{i}_{t}"))
                if t == 0:
                    self.formula.append([ld[i][t]])
                else:
                    if i > 1:
                        self.formula.append([-ld[i - 1][t - 1], -self.v[i], ld[i][t]])
                        if t < len(ld[i-1]):
                            # Carry over
                            self.formula.append([-ld[i-1][t], ld[i][t]])
                            # Increment if leaf
                            # i == 1 cannot be a leaf, as it is the root
                            self.formula.append([-ld[i][t], ld[i-1][t], ld[i-1][t-1]])
                            self.formula.append([-ld[i][t], ld[i-1][t], self.v[i]])
                        else:
                            self.formula.append([-ld[i][t], ld[i - 1][t - 1]])
                            self.formula.append([-ld[i][t], self.v[i]])
                    # Use bound
                    if 2*(i-t+1) <= num_nodes:
                        self.formula.append([-ld[i][t], -self.left[i][2*(i-t+1)]])
                    if 2*(i-t+1)+1 <= num_nodes:
                        self.formula.append([-ld[i][t], -self.right[i][2*(i-t+1)+1]])

        tau = [None]
        for i in range(1, num_nodes+1):
            tau.append([])
            for t in range(0, i+1):
                tau[i].append(self.pool.id(f"tau{i}_{t}"))
                if t == 0:
                    self.formula.append([tau[i][t]])
                else:
                    if i > 1:
                        # Increment
                        self.formula.append([-tau[i - 1][t - 1], self.v[i], -tau[i][t]])
                        if t < len(tau[i-1]):
                            # Carry over
                            self.formula.append([-tau[i-1][t], tau[i][t]])

                            # Reverse equivalence
                            self.formula.append([-tau[i][t], tau[i-1][t], tau[i-1][t-1]])
                            self.formula.append([-tau[i][t], tau[i-1][t], -self.v[i]])
                        else:
                            # Reverse equivalence
                            self.formula.append([-tau[i][t], tau[i - 1][t - 1]])
                            self.formula.append([-tau[i][t], -self.v[i]])

                if t > (i//2) + (i % 2): # i/2 rounded up
                    # Use bound
                    if num_nodes >= 2*(t - 1) > i:
                        self.formula.append([-tau[i][t], -self.left[i][2*(t-1)]])
                    if i < 2*t-1 <= num_nodes:
                        self.formula.append([-tau[i][t], -self.right[i][2*t-1]])

        # root is the first non-leaf
        #self.formula.append([tau[1][1]])

    def encode(self, instance, num_nodes, improve=True):
        self.formula = CNF()
        classes = set()
        for e in instance.examples:
            classes.add(e.cls)

        c_vars = len(bin(len(classes) - 1)) - 2  # "easier" than log_2
        classes = list(classes)  # Give classes an order
        classes.sort()

        self.class_map = {}

        for i in range(0, len(classes)):
            self.class_map[classes[i]] = []
            for c_v in bin(i)[2:][::-1]:
                if c_v == "1":
                    self.class_map[classes[i]].append(True)
                else:
                    self.class_map[classes[i]].append(False)

            while len(self.class_map[classes[i]]) < c_vars:
                self.class_map[classes[i]].append(False)

        self.init_vars(instance, num_nodes, c_vars)
        self.encode_tree_structure(instance, num_nodes)
        self.encode_discriminating(instance, num_nodes)
        self.encode_feature(instance, num_nodes)
        self.encode_examples(instance, num_nodes, self.class_map)
        if improve:
            self.improve(num_nodes)
        # TODO: Check why this causes UNSAT
        #self.uniquify_classes(instance, num_nodes, self.class_map)

    def uniquify_classes(self, instance, num_nodes, class_map):
        # Disallow wrong labels
        # Forbid non-existing classes
        # Generate all class identifiers
        c_vars = len(next(iter(class_map.values())))
        for c_c in itertools.product([True, False], repeat=c_vars):
            # Check if identifier is used
            exists = False
            for c_v in self.class_map.values():
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
                    clause = [-self.v[i]]
                    for c in range(0, c_vars):
                        clause.append(-self.c[i][c] if c_c[c] else self.c[i][c])
                    self.formula.append([*clause])

        # Avoid for specific classes to be used only once
        for i in range(1, num_nodes + 1):
            for ck, cv in class_map.items():
                if ck < 0:
                    continue
                for j in range(i + 1, num_nodes + 1):
                    clause = [-self.v[i], -self.v[j]]
                    for c in range(0, len(cv)):
                        modifier = -1 if cv[c] else 1

                        clause.append(modifier * self.c[i][c])
                        clause.append(modifier * self.c[j][c])
                    self.formula.append([*clause])

    def run(self, instance, solver, start_bound=3, timeout=0, ub=maxsize):
        c_bound = start_bound
        best_model = None
        lb = 0

        while lb < ub:
            print(f"Running {c_bound}")
            with solver() as slv:
                self.encode(instance, c_bound)
                slv.append_formula(self.formula)
                if timeout == 0:
                    solved = slv.solve()
                else:
                    def interrupt(s):
                        s.interrupt()

                    timer = Timer(timeout, interrupt, [slv])
                    timer.start()
                    solved = slv.solve_limited(expect_interrupt=True)

                if solved:
                    model = {abs(x): x > 0 for x in slv.get_model()}
                    best_model = self.decode(model, instance, c_bound)
                    ub = c_bound
                    c_bound -= 2
                else:
                    c_bound += 2
                    lb = c_bound

        return best_model

    def decode(self, model, instance, num_nodes):
        # TODO: This could be faster, but for debugging purposes, check for consistency
        tree = DecisionTree(instance.num_features, num_nodes)
        # Set root
        for r in range(1, instance.num_features + 1):
            if model[self.a[r][1]]:
                if tree.root is None:
                    tree.set_root(r)
                else:
                    print(f"ERROR: Duplicate feature for root, set feature {tree.root.feature}, current {r}")

        if tree.root is None:
            print(f"ERROR: No feature found for root")

        # Add other nodes
        for j in range(2, num_nodes + 1):
            is_leaf = model[self.v[j]]

            parent = None
            for i in SizeNarodytska.pr(j):
                if model[self.p[j][i]]:
                    if parent is None:
                        parent = i
                    else:
                        print(f"ERROR: Double parent for {j}, set {parent} also found {i}")
            if parent is None:
                print(f"ERROR: No parent found for {j}")
                raise

            feature = None
            if (j % 2 == 0 and not model[self.left[parent][j]]) or (j % 2 == 1 and not model[self.right[parent][j]]):
                print(f"ERROR: Parent - Child relationship mismatch, parent {parent}, child {j}")

            if not is_leaf:
                for r in range(1, instance.num_features + 1):
                    if model[self.a[r][j]]:
                        if feature is None:
                            feature = r
                        else:
                            print(f"ERROR: Duplicate feature for {j}, set feature {feature}, current {r}")
                tree.add_node(j, parent, feature, j % 2 == 0)
            else:
                c_c = None
                for k, v in self.class_map.items():
                    failed = False
                    for c in range(0, len(v)):
                        if model[self.c[j][c]] != v[c]:
                            failed = True
                    if not failed:
                        c_c = k
                assert c_c is not None
                tree.add_leaf(j, parent, j % 2 == 0, c_c)

        self.check_consistency(model, instance, num_nodes, tree)
        return tree

    def check_consistency(self, model, instance, num_nodes, tree):
        # Check left, right vars
        for j in range(2, num_nodes + 1):
            cnt = 0
            for i in SizeNarodytska.pr(j):
                if j % 2 == 0 and model[self.left[i][j]]:
                    cnt += 1
                elif j % 2 == 1 and model[self.right[i][j]]:
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
                values = []
                path = [node]
                cp = node
                # trace path from leaf to root
                while cp != 1:
                    path.append(prev[cp])
                    features.append(tree.nodes[prev[cp]].feature)
                    values.append(tree.nodes[prev[cp]].left.id == cp)
                    cp = prev[cp]
                path.pop()
                path.reverse()
                features.reverse()
                values.reverse()

                # Now verify the model
                for i in range(0, len(path)):
                    feat = features[i]
                    cnode = path[i]
                    d1val = values[i]
                    d0val = not values[i]

                    for j in range(i, len(path)):
                        if not tree.nodes[path[j]].is_leaf and tree.nodes[path[j]].feature == feat:
                            print(f"ERROR duplicate feature {feat} in nodes {cnode} and {path[j]}")
                        if not model[self.u[feat][path[j]]]:
                            print(f"ERROR u for feature {feat} not set for node {path[j]}")
                        if model[self.d0[feat][path[j]]] != d0val:
                            print(f"ERROR d0 value wrong, feature {feat} at node {path[j]} is leaf: {tree.nodes[path[j]].is_leaf}")
                        if model[self.d1[feat][path[j]]] != d1val:
                            print(f"ERROR d1 value wrong, feature {feat} at node {path[j]} is leaf: {tree.nodes[path[j]].is_leaf}")

    @staticmethod
    def new_bound(tree, instance):
        if tree is None:
            return 3

        return min(len(tree.nodes) - 1, 2 * 2**instance.num_features - 1)

    @staticmethod
    def lb():
        return 3

    @staticmethod
    def max_instances(num_features, limit):
        if num_features < 50:
            return 100
        if num_features < 100:
            return 70
        return 50
