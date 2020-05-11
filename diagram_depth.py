import base_encoding
from decision_tree import DecisionDiagram


class DiagramDepthEncoding(base_encoding.BaseEncoding):
    def __init__(self, stream):
        base_encoding.BaseEncoding.__init__(self, stream)
        self.g = None
        self.d = None

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

    def encode(self, instance, depth):
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
                            self.add_clause(-self.g[i][j][dl], -self.g[j][k][dl], self.g[i][k][dl])
                            self.add_clause(-self.g[i][j][dl], -self.g[i][k][dl], self.g[j][k][dl])
                            self.add_clause(-self.g[i][k][dl], -self.g[j][k][dl], self.g[i][j][dl])

        # Verify that at last level, the partitioning is by class
        for i in range(0, len(instance.examples)):
            for j in range(i + 1, len(instance.examples)):
                if instance.examples[i].cls == instance.examples[j].cls:
                    self.add_clause(self.g[i][j][depth])
                else:
                    self.add_clause(-self.g[i][j][depth])

        # Verify that the examples are partitioned correctly
        for i in range(0, len(instance.examples)):
            for j in range(i + 1, len(instance.examples)):
                for dl in range(0, depth):
                    for f in range(1, instance.num_features+1):
                        if instance.examples[i].features[f] == instance.examples[j].features[f]:
                            self.add_clause(-self.g[i][j][dl], -self.d[i][dl][f], self.g[i][j][dl+1])
                        else:
                            self.add_clause(-self.g[i][j][dl], -self.d[i][dl][f], -self.g[i][j][dl+1])

        # Verify that group cannot merge
        # for i in range(0, len(instance.examples)):
        #     for j in range(i + 1, len(instance.examples)):
        #         for dl in range(0, depth):
        #             self.add_clause(self.g[i][j][dl], -self.g[i][j][dl + 1])

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
                    for f2 in range(f + 1, instance.num_features + 1):
                        self.add_clause(-self.d[i][dl][f], -self.d[i][dl][f2])
                self.add_clause(*clause)

        self.write_header(instance)

    def decode(self, model, instance, depth):
        tree = DecisionDiagram(instance.num_features, 2**(depth+1) - 1)

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

        # Start with pos, neg groups
        node_id = 2**(depth+1) - 1
        groups = {
            node_id: ([i for i in range(0, len(instance.examples)) if not instance.examples[i].cls], None, None, None),
            node_id-1: ([i for i in range(0, len(instance.examples)) if instance.examples[i].cls], None, None, None)
        }
        node_id -= 2

        for dl in range(depth - 1, -1, -1):
            new_groups = {}
            while groups:
                p = next(iter(groups))
                g, lc, rc, feat = groups.pop(p)
                g.sort(reverse=True)

                # Not a leaf
                if not (lc is None and rc is None):
                    # First to options, group is not split, so skip node
                    if lc is None and depth > 0:
                        p = rc
                    elif rc is None and depth > 0:
                        p = lc
                    else:
                        # This is an edge case happening if all examples have the same class
                        if rc is None:
                            rc = lc
                        if lc is None:
                            lc = rc
                        # edge case end
                        tree.add_node(p, feat, lc, rc)
                        tree.root = tree.nodes[p]  # Root is the last added node

                # Separate g into groups
                while g:
                    n = g.pop()
                    found = False
                    feat = find_feature(n, dl)
                    feat_val = instance.examples[n].features[feat]

                    for c_id, (c_g, c_lc, c_rc, c_ft) in new_groups.items():
                        n1 = c_g[0] if c_g[0] < n else n
                        n2 = n if c_g[0] < n else c_g[0]
                        if model[self.g[n1][n2][dl]]:
                            if c_ft != feat:
                                print("ERROR: Two items belonging to the same group are split on a different feature")
                            if feat_val:
                                if c_lc is None:
                                    new_groups[c_id] = (c_g, p, c_rc, c_ft)
                                elif c_lc != p:
                                    print("ERROR: Left child already set, but is a different left child")
                            else:
                                if c_rc is None:
                                    new_groups[c_id] = (c_g, c_lc, p, c_ft)
                                elif c_rc != p:
                                    print("ERROR: Right child already set, but is a different right child")
                            c_g.append(n)

                            found = True

                    if not found:
                        if feat_val:
                            new_groups[node_id] = ([n], p, None, feat)
                        else:
                            new_groups[node_id] = ([n], None, p, feat)
                        node_id -= 1

            groups = new_groups
            for i in range(0, len(instance.examples)):
                cnt = 0
                for g, _, _, _ in groups.values():
                    if i in g:
                        cnt += 1
                if cnt != 1:
                    print("ERROR: Each example should occur in exactly one group")

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
