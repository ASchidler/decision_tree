import base_encoding
from decision_tree import DecisionDiagram


class DecisionDiagramEncoding(base_encoding.BaseEncoding):
    def __init__(self, stream):
        base_encoding.BaseEncoding.__init__(self, stream)
        self.ord = None
        self.left = None
        self.right = None
        self.feature = None

    def init_var(self, instance, num_nodes):
        self.left = [{} for _ in range(0, num_nodes + 1)]
        self.right = [{} for _ in range(0, num_nodes + 1)]

        # Successors. The last two nodes are leafs for 0 and 1
        for i in range(1, num_nodes + 1 - 2):
            for j in range(i+1, num_nodes+1):
                self.left[i][j] = self.add_var()
                self.right[i][j] = self.add_var()

        self.feature = [None]
        for r in range(1, instance.num_features + 1):
            self.feature.append([None])
            for j in range(1, num_nodes + 1 - 2):
                self.feature[r].append(self.add_var())

    def encode_diagram(self, instance, num_nodes):
        # Each node must have a right and left successor, except the last two
        for i in range(1, num_nodes+1 - 2):
            clause1 = []
            clause2 = []
            for j in range(i + 1, num_nodes + 1):
                clause1.append(self.left[i][j])
                clause2.append(self.right[i][j])

                # Not more than one
                for j2 in range(j + 1, num_nodes + 1):
                    self.add_clause(-self.left[i][j], -self.left[i][j2])
                    self.add_clause(-self.right[i][j], -self.right[i][j2])
            self.add_clause(*clause1)
            self.add_clause(*clause2)

    def encode_features(self, instance, num_nodes):
        # Each node, except the leafs, has assigned exactly one feature
        for i in range(1, num_nodes + 1 - 2):
            clause = []

            for r in range(1, instance.num_features + 1):
                clause.append(self.feature[r][i])
                for r2 in range(r + 1, instance.num_features + 1):
                    self.add_clause(-self.feature[r][i], -self.feature[r2][i])
            self.add_clause(*clause)

    def encode_examples(self, instance, num_nodes):
        # evars express that the node is reachable for the current example

        for e in instance.examples:
            fvar = [self.add_var() if i > 0 else None for i in range(0, num_nodes + 1)]
            pvar = [self.add_var() if i > 0 else None for i in range(0, num_nodes + 1)]

            for i in range(1, num_nodes + 1 - 2):
                for r in range(1, instance.num_features + 1):
                    if e.features[r]:
                        self.add_clause(-self.feature[r][i], fvar[i])
                    else:
                        self.add_clause(-self.feature[r][i], -fvar[i])

            # Place forbidden marker on the class that must not be reachable
            if e.cls:
                self.add_clause(-pvar[-1])
            else:
                self.add_clause(-pvar[-2])

            # Root is always reachable
            self.add_clause(pvar[1])

            # Propagate the marker
            for i in range(1, num_nodes + 1 - 2):
                for j in range(i + 1, num_nodes + 1):
                    self.add_clause(-self.left[i][j], -pvar[i], -fvar[i], pvar[j])
                    self.add_clause(-self.right[i][j], -pvar[i], fvar[i], pvar[j])

    def encode_examples2(self, instance, num_nodes):

        for e in instance.examples:
            fvar = [self.add_var() if i > 0 else None for i in range(0, num_nodes + 1)]

            # Place forbidden marker on the class that must not be reachable
            # Root is always reachable
            self.add_clause(fvar[1])
            if e.cls:
                self.add_clause(-fvar[-1])
                self.add_clause(fvar[-2])
            else:
                self.add_clause(fvar[-1])
                self.add_clause(-fvar[-2])

            for i in range(1, num_nodes + 1 - 2):
                for j in range(i + 1, num_nodes + 1):
                    for r in range(1, instance.num_features + 1):
                        if e.features[r]:
                            self.add_clause(-self.feature[r][i], -self.left[i][j], -fvar[i], fvar[j])
                        else:
                            self.add_clause(-self.feature[r][i], -self.right[i][j], -fvar[i], fvar[j])

    def encode(self, instance, num_nodes):
        self.init_var(instance, num_nodes)
        self.encode_diagram(instance, num_nodes)
        self.encode_features(instance, num_nodes)
        self.encode_examples(instance, num_nodes)

    def decode(self, model, instance, num_nodes):
        bdd = DecisionDiagram(instance.num_features, num_nodes)
        for i in range(num_nodes-2, 0, -1):
            feature = None

            for r in range(1, instance.num_features + 1):
                if model[self.feature[r][i]]:
                    if feature is not None:
                        print(f"ERROR: Node {i} already has feature {feature} assigned, but feature {r} is also set")
                    else:
                        feature = r
            if feature is None:
                print(f"ERROR: No feature found for node {i}")

            lnode = None
            for j in range(i+1, num_nodes + 1):
                if model[self.left[i][j]]:
                    if lnode is not None:
                        print(f"ERROR: Multiple left nodes found for {i}: {lnode} and {j}")
                    else:
                        lnode = j

            rnode = None
            for j in range(i + 1, num_nodes + 1):
                if model[self.right[i][j]]:
                    if rnode is not None:
                        print(f"ERROR: Multiple left nodes found for {i}: {rnode} and {j}")
                    else:
                        rnode = j

            bdd.add_node(i, feature, lnode, rnode)
            if i == 1:
                bdd.root = bdd.nodes[1]

        return bdd

    @staticmethod
    def new_bound(tree, instance):
        if tree is None:
            return 3

        return len(tree.nodes) - 1

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
