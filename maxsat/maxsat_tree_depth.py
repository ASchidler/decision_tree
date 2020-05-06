import base_encoding
from decision_tree import DecisionTree
import math


class TreeDepthEncoding(base_encoding.BaseEncoding):
    def __init__(self, stream):
        base_encoding.BaseEncoding.__init__(self, stream)
        self.g = None
        self.d = None
        self.c = None

    def init_vars(self, instance, depth):
        self.d = []
        for i in range(0, len(instance.examples)):
            self.d.append([])
            for dl in range(0, depth):
                self.d[i].append([None])
                for f in range(1, instance.num_features + 1):
                    self.d[i][dl].append(self.add_var())

        self.g = [{} for _ in range(0, len(instance.examples))]
        for i in range(0, len(instance.examples)):
            for j in range(i + 1, len(instance.examples)):
                self.g[i][j] = [self.add_var() for _ in range(0, depth + 1)]

        self.c = [self.add_var() if i > 1 else None for i in range(0, depth + 1)]

    def encode(self, instance):
        # Maximum depth is either one for each feature, or enough such that all examples have a unique leaf.
        depth = min(math.ceil(math.log2(len(instance.examples))), instance.num_features + 1)
        self.init_vars(instance, depth)

        # Add level 0, all examples are in the same group
        for i in range(0, len(instance.examples)):
            for j in range(i + 1, len(instance.examples)):
                self.add_clause(self.g[i][j][0])

        # Transitivity
        for i in range(0, len(instance.examples)):
            for j in range(i + 1, len(instance.examples)):
                for dl in range(1, depth + 1):
                    for k in range(j + 1, len(instance.examples)):
                        if i != k and j != k:
                            # TODO: is this right or does k have to iterate over all values?
                            # Consistency check says it works...
                            self.add_clause(-self.g[i][k][dl], -self.g[j][k][dl], self.g[i][k][dl])

        # Verify that at last level, the partitioning is by class
        for i in range(0, len(instance.examples)):
            for j in range(i + 1, len(instance.examples)):
                if instance.examples[i].cls != instance.examples[j].cls:
                    self.add_clause(-self.g[i][j][depth])

        # Verify that the examples are partitioned correctly
        for i in range(0, len(instance.examples)):
            for j in range(i + 1, len(instance.examples)):
                for dl in range(0, depth):
                    for f in range(1, instance.num_features+1):
                        if instance.examples[i].features[f] == instance.examples[j].features[f]:
                            self.add_clause(-self.g[i][j][dl], -self.d[i][dl][f], self.g[i][j][dl+1])
                        else:
                            self.add_clause(-self.d[i][dl][f], -self.g[i][j][dl + 1])

        # Verify that group cannot merge
        for i in range(0, len(instance.examples)):
            for j in range(i + 1, len(instance.examples)):
                for dl in range(0, depth):
                    self.add_clause(self.g[i][j][dl], -self.g[i][j][dl + 1])

        # Verify that d is consistent
        for i in range(0, len(instance.examples)):
            for j in range(i + 1, len(instance.examples)):
                for dl in range(0, depth):
                    for f in range(1, instance.num_features+1):
                        self.add_clause(-self.g[i][j][dl], -self.d[i][dl][f], self.d[j][dl][f])

        # One feature per level and group
        for i in range(0, len(instance.examples)):
            for dl in range(0, depth):
                clause = []
                for f in range(1, instance.num_features + 1):
                    clause.append(self.d[i][dl][f])
                    for f2 in range(f + 1, instance.num_features):
                        self.add_clause(-self.d[i][dl][f], -self.d[i][dl][f2])
                self.add_clause(*clause)

    def decode(self, model, instance, depth):
        tree = DecisionTree(instance.num_features, 2**(depth+1) - 1)

        # Root
        def find_feature(ce, cdl):
            ce_feature = None
            for cf in range(1, instance.num_features+1):
                if model[self.d[ce][cdl][cf]]:
                    if ce_feature is None:
                        ce_feature = cf
                    else:
                        print(f"ERROR double feature {cf} and {ce_feature} for experiment {ce}, at level {cdl}.")
            if ce_feature is None:
                print(f"ERROR no feature for {ce} at level {cdl}.")
            return ce_feature

        node = 2
        feature = find_feature(0, 0)
        tree.set_root(feature)
        groups = {1: ([i for i in range(0, len(instance.examples))], None, None)}

        for dl in range(0, depth+1):
            new_groups = {}
            while groups:
                p = next(iter(groups))
                g, pol, pp = groups.pop(p)
                g1 = []
                g2 = []
                g.reverse()  # Process in increasing order when popping
                n = g.pop()  # Use smallest element as reference
                g1.append(n)
                # Separate g into two groups, based in n
                while g:
                    n2 = g.pop()
                    if model[self.g[n][n2][dl]]:
                        g1.append(n2)
                    else:
                        g2.append(n2)

                # Check if the decision split the group
                if len(g2) == 0:
                    new_groups[p] = (g1, pol, pp)
                else:
                    if p > 1:
                        f = find_feature(n, dl-1)
                        tree.add_node(p, pp, f, pol)

                    polarity = instance.examples[n].features[tree.nodes[p].feature]
                    new_groups[node] = (g1, polarity, p)
                    node += 1
                    new_groups[node] = (g2, not polarity, p)
                    node += 1

                    # Consistency
                    for cg in [g1, g2]:
                        f = tree.nodes[p].feature
                        for i in range(0, len(cg)):
                            for j in range(i+1, len(cg)):
                                if not model[self.g[cg[i]][cg[j]][dl]]:
                                    print("ERROR: Group mismatch")
                                if instance.examples[cg[i]].features[f] != instance.examples[cg[j]].features[f]:
                                    print("ERROR: Feature values inconsistency")
                        n = cg[0]
                        for i in range(n+1, len(instance.examples)):
                            if model[self.g[n][i][dl]] and i not in cg:
                                print("ERROR: Element missing")

            groups = new_groups

        # Find leafs
        for p, (group, pol, pp) in groups.items():
            cls = instance.examples[group[0]].cls
            for e in group:
                if instance.examples[e].cls != cls:
                    print("Error, non matching class in last group!")
            tree.add_leaf(p, pp, pol, cls)

        # Simplify
        def simplify(node):
            if node.is_leaf:
                return

            changed = True
            while changed:
                changed = False

                if not node.left.is_leaf and node.feature == node.left.feature:
                    node.left = node.left.left
                    changed = True
                if not node.right.is_leaf and node.feature == node.right.feature:
                    node.right = node.right.right
                    changed = True

            simplify(node.left)
            simplify(node.right)

        simplify(tree.root)
        return tree

    def check_consistency(self, model, instance, num_nodes, tree):
        pass

    @staticmethod
    def new_bound(tree, instance):
        if tree is None:
            return 1

        def dfs_find(node, level):
            if node.is_leaf:
                return level
            else:
                return max(dfs_find(node.left, level + 1), dfs_find(node.right, level + 1))

        return dfs_find(tree.root, 0)

    @staticmethod
    def lb():
        return 1