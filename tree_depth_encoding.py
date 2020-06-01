import base_encoding
from decision_tree import DecisionTree, NonBinaryTree
import math

class TreeDepthEncoding(base_encoding.BaseEncoding):
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
        # for i in range(0, len(instance.examples)):
        #     for j in range(i + 1, len(instance.examples)):
        #         for dl in range(1, depth + 1):
        #             for k in range(j + 1, len(instance.examples)):
        #                 if i != k and j != k:
        #                     self.add_clause(-self.g[i][j][dl], -self.g[j][k][dl], self.g[i][k][dl])
        #                     self.add_clause(-self.g[i][j][dl], -self.g[i][k][dl], self.g[j][k][dl])
        #                     self.add_clause(-self.g[i][k][dl], -self.g[j][k][dl], self.g[i][j][dl])

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
                    for f2 in range(f + 1, instance.num_features + 1):
                        self.add_clause(-self.d[i][dl][f], -self.d[i][dl][f2])
                self.add_clause(*clause)

        self.write_header(instance)

    def decode_nonbinary(self, model, instance, depth):
        tree = NonBinaryTree()

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

        def df_tree(grp, parent, d):
            if d == depth:
                cls = grp[0][1].cls
                for _, e in grp:
                    if e.cls != cls:
                        print(f"Error, double cls in leaf group {cls}, {e.cls}")
                tree.add_leaf(parent, grp[0][1].features[parent.feature], cls)
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

                    if model[self.g[u][v][d+1]]:
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
                val = None if parent is None else grp[0][1].features[parent.feature]
                n_n = tree.add_node(parent, val, f)
                for ng in new_grps:
                   df_tree(ng, n_n, d+1)
            else:
                df_tree(new_grps[0], parent, d+1)

        df_tree(list(enumerate(instance.examples)), None, 0)
        return tree

    def decode(self, model, instance, depth):
        if not instance.is_binary():
            return self.decode_nonbinary(model, instance, depth)

        tree = DecisionTree(instance.num_features, 1)

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

                    if model[self.g[u][v][d+1]]:
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
    def max_instances(num_features, limit):
        if num_features < 20:
            return 50
        if num_features < 35:
            return 40
        return 25

    @staticmethod
    def lb():
        return 1