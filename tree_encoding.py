import base_encoding
from decision_tree import DecisionTree


class TreeEncoding(base_encoding.BaseEncoding):
    def __init__(self, stream):
        base_encoding.BaseEncoding.__init__(self, stream)
        self.v = None
        self.left = None
        self.right = None
        self.p = None
        self.a = None
        self.u = None
        self.d0 = None
        self.d1 = None
        self.c = None

    def init_vars(self, instance, num_nodes):
        # First the tree structure
        self.v = []
        self.v.append(None) # Avoid 0 index
        for _ in range(1, num_nodes + 1):
            self.v.append(self.add_var())

        self.right = [{} for _ in range(0, num_nodes + 1)]
        self.left = [{} for _ in range(0, num_nodes + 1)]
        for i in range(1, num_nodes + 1):
            for j in range(i+1, min(2 * i + 2, num_nodes + 1)):
                if j % 2 == 0:
                    self.left[i][j] = self.add_var()
                else:
                    self.right[i][j] = self.add_var()

        self.p = [{} for _ in range(0, num_nodes + 1)]
        for j in range(2, num_nodes + 1, 2): # Starts from 2 not 1
            for i in range(j // 2, j):
                self.p[j][i] = self.add_var()
                self.p[j+1][i] = self.add_var()

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
                self.a[i].append(self.add_var())
                self.u[i].append(self.add_var())
                self.d0[i].append(self.add_var())
                self.d1[i].append(self.add_var())
        self.c = [self.add_var() if i > 0 else None for i in range(0, num_nodes + 1)]

    @staticmethod
    def lr(i, mx):
        if i % 2 == 0:
            return range(i + 2, min(2 * i, mx - 1) + 1, 2)
        else:
            return range(i + 1, min(2 * i, mx - 1) + 1, 2)

    @staticmethod
    def rr(i, mx):
        if i % 2 == 0:
            return range(i + 3, min(2 * i, mx) + 1, 2)
        else:
            return range(i + 2, min(2 * i, mx) + 1, 2)

    @staticmethod
    def pr(j):
        if j % 2 == 0:
            return range(max(1, j // 2), j)
        else:
            return range(max(1, (j - 1) // 2), j-1)

    def encode_tree_structure(self, instance, num_nodes):
        # root is not a leaf
        self.add_clause(-self.v[1])

        for i in range(1, num_nodes + 1):
            for j in TreeEncoding.lr(i, num_nodes):
                # Leafs have no children
                self.add_clause(-self.v[i], -self.left[i][j])
                # children are consecutively numbered
                self.add_clause(-self.left[i][j], self.right[i][j + 1])
                self.add_clause(self.left[i][j], -self.right[i][j + 1])

        # Enforce parent child relationship
        for i in range(1, num_nodes + 1):
            for j in TreeEncoding.lr(i, num_nodes):
                self.add_clause(-self.p[j][i], self.left[i][j])
                self.add_clause(self.p[j][i], -self.left[i][j])
            for j in TreeEncoding.rr(i, num_nodes):
                self.add_clause(-self.p[j][i], self.right[i][j])
                self.add_clause(self.p[j][i], -self.right[i][j])

        # Cardinality constraint
        # Each non leaf must have exactly one left child
        for i in range(1, num_nodes + 1):
            # First must have a child
            nodes = []
            for j in TreeEncoding.lr(i, num_nodes):
                nodes.append(self.left[i][j])
            self.add_clause(self.v[i], *nodes)
            # Next, not more than one
            for j1 in TreeEncoding.lr(i, num_nodes):
                for j2 in TreeEncoding.lr(i, num_nodes):
                    if j2 > j1:
                        self.add_clause(self.v[i], -self.left[i][j1], -self.left[i][j2])

        # Each non-root must have exactly one parent
        for j in range(2, num_nodes + 1, 2):
            clause1 = []
            clause2 = []
            for i in range(j//2, j):
                clause1.append(self.p[j][i])
                clause2.append(self.p[j+1][i])
            self.add_clause(*clause1)
            self.add_clause(*clause2)

            for i1 in range(j//2, j):
                for i2 in range(i1 + 1, j):
                    self.add_clause(-self.p[j][i1], -self.p[j][i2])
                    self.add_clause(-self.p[j+1][i1], -self.p[j+1][i2])

    def encode_discriminating(self, instance, num_nodes):
        for r in range(1, instance.num_features + 1):
            self.add_clause(-self.d0[r][1])
            self.add_clause(-self.d1[r][1])

        # Discriminating features
        for j in range(2, num_nodes + 1, 2):
            for r in range(1, instance.num_features + 1):
                for direction in [False, True]:
                    jpathl = self.d1[r][j] if direction else self.d0[r][j]
                    jpathr = self.d1[r][j+1] if direction else self.d0[r][j+1]
                    for i in TreeEncoding.pr(j):
                        ipath = self.d1[r][i] if direction else self.d0[r][i]

                        # Children inherit from the parent
                        self.add_clause(-self.left[i][j], -ipath, jpathl)
                        self.add_clause(-self.right[i][j+1], -ipath, jpathr)

                        if direction:
                            # The current node discriminates
                            self.add_clause(-self.left[i][j], -self.a[r][i], jpathl)
                            # Other side of the implication
                            self.add_clause(-jpathl, -self.left[i][j], ipath, self.a[r][i])
                            self.add_clause(-jpathr, -self.right[i][j+1], ipath)
                        else:
                            self.add_clause(-self.right[i][j+1], -self.a[r][i], jpathr)
                            # Other side of the implication
                            self.add_clause(-jpathl, -self.left[i][j], ipath)
                            self.add_clause(-jpathr, -self.right[i][j + 1], ipath, self.a[r][i])

    def encode_feature(self, instance, num_nodes):
        # Feature assignment
        # u means that the feature already occurred in the current branch
        for r in range(1, instance.num_features + 1):
            for j in range(1, num_nodes + 1):
                clause = [-self.u[r][j], self.a[r][j]]

                for i in TreeEncoding.pr(j):
                    # If u is true for the parent, the feature must not be used by any child
                    self.add_clause(-self.u[r][i], -self.p[j][i], -self.a[r][j])

                    v1 = self.add_auxiliary(self.u[r][i], self.p[j][i])
                    clause.append(v1)
                    self.add_clause(-self.a[r][j], self.u[r][j])
                    #self.add_clause(-self.u[r][i], -self.p[j][i], self.u[r][j])
                    self.add_clause(-v1, self.u[r][j])
                self.add_clause(*clause)

        # Leafs have no feature
        for r in range(1, instance.num_features + 1):
            for j in range(1, num_nodes + 1):
                self.add_clause(-self.v[j], -self.a[r][j])

        # Non-Leafs have exactly one feature
        for j in range(1, num_nodes + 1):
            clause = [self.v[j]]
            for r in range(1, instance.num_features + 1):
                clause.append(self.a[r][j])
                for r2 in range(r + 1, instance.num_features + 1):
                    self.add_clause(-self.a[r][j], -self.a[r2][j])
            self.add_clause(*clause)

    def encode_examples(self, instance, num_nodes):
        for e in instance.examples:
            for j in range(1, num_nodes + 1):
                # If the class of the leaf differs from the class of the example, at least one
                # node on the way must discriminate against the example, otherwise the example
                # could be classified wrong
                clause = [-self.v[j]]
                clause.append(self.c[j] if e.cls else -self.c[j])
                for r in range(1, instance.num_features):
                    clause.append(self.d0[r][j] if e.features[r] else self.d1[r][j])
                self.add_clause(*clause)

    def encode(self, instance, num_nodes):
        self.init_vars(instance, num_nodes)
        self.encode_tree_structure(instance, num_nodes)
        self.encode_discriminating(instance, num_nodes)
        self.encode_feature(instance, num_nodes)
        self.encode_examples(instance, num_nodes)

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
        # TODO: Theoretically parent only has to go from j/2 to j-1
        for j in range(2, num_nodes + 1):
            is_leaf = model[self.v[j]]

            parent = None
            for i in TreeEncoding.pr(j):
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
                    if model[self.a[r][1]]:
                        if feature is None:
                            feature = r
                        else:
                            print(f"ERROR: Duplicate feature for {j}, set feature {feature}, current {r}")
                tree.add_node(j, parent, feature, j % 2 == 0)
            else:
                tree.add_leaf(j, parent, j % 2 == 0, model[self.c[j]])

        return tree
