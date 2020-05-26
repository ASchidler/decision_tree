import base_encoding
from decision_tree import DecisionTree


class AAAIEncoding(base_encoding.BaseEncoding):
    def __init__(self, stream):
        base_encoding.BaseEncoding.__init__(self, stream)
        self.x = None
        self.f = None
        self.c = None

    def init_var(self, instance, limit):
        self.x = [[] for _ in range(0, len(instance.examples))]
        for xl in self.x:
            for _ in range(0, limit):
                xl.append(self.add_var())

        self.f = [[None] for _ in range(0, 2**limit)]
        # index starting with 1
        for i in range(1, 2**limit):
            for _ in range(1, instance.num_features + 1):
                self.f[i].append(self.add_var())

        self.c = [self.add_var() for _ in range(0, 2**limit)]

    def encode(self, instance, limit):
        self.init_var(instance, limit)

        # each node has a feature
        for i in range(1, 2**limit):
            clause = []
            for f1 in range(1, instance.num_features + 1):
                clause.append(self.f[i][f1])
                for f2 in range(f1+1, instance.num_features + 1):
                    self.add_clause(-self.f[i][f1], -self.f[i][f2])
            self.add_clause(*clause)

        for i in range(0, len(instance.examples)):
            self.alg1(instance, i, limit, 0, 1, list())
            self.alg2(instance, i, limit, 0, 1, list())

    def alg1(self, instance, e_idx, limit, lvl, q, clause):
        if lvl == limit:
            return

        example = instance.examples[e_idx]
        for f in range(1, instance.num_features + 1):
            if not example.features[f]:
                self.add_clause(*clause, -self.x[e_idx][lvl], -self.f[q][f])
        n_cl = list(clause)
        n_cl.append(-self.x[e_idx][lvl])
        self.alg1(instance, e_idx, limit, lvl+1, 2 * q + 1, n_cl)

        for f in range(1, instance.num_features + 1):
            if example.features[f]:
                self.add_clause(*clause, self.x[e_idx][lvl], -self.f[q][f])
        n_cl2 = list(clause)
        n_cl2.append(self.x[e_idx][lvl])
        self.alg1(instance, e_idx, limit, lvl+1, 2*q, n_cl2)

    def alg2(self, instance, e_idx, limit, lvl, q, clause):
        if lvl == limit:
            if instance.examples[e_idx].cls:
                self.add_clause(*clause, self.c[q - 2**limit])
            else:
                self.add_clause(*clause, -self.c[q - 2**limit])
        else:
            n_cl = list(clause)
            n_cl.append(self.x[e_idx][lvl])
            n_cl2 = list(clause)
            n_cl2.append(-self.x[e_idx][lvl])
            self.alg2(instance, e_idx, limit, lvl+1, 2*q, n_cl)
            self.alg2(instance, e_idx, limit, lvl+1, 2*q+1, n_cl2)

    def decode(self, model, instance, limit):
        num_leafs = 2**limit
        tree = DecisionTree(instance.num_features, 2 * num_leafs - 1)
        # Find features
        for i in range(1, num_leafs):
            f_found = False
            for f in range(1, instance.num_features+1):
                if model[self.f[i][f]]:
                    if f_found:
                        print(f"ERROR: double features found for node {i}, features {f} and {tree.nodes[i].feature}")
                    else:
                        if i == 1:
                            tree.set_root(f)
                        else:
                            tree.add_node(i, i//2, f, i % 2 == 1)

        for i in range(0, num_leafs):
            tree.add_leaf(num_leafs + i, (num_leafs + i)//2, i % 2 == 1, model[self.c[i]])

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
