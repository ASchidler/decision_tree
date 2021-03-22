from decision_tree import DecisionTree, NonBinaryTree
import itertools
from pysat.formula import IDPool, CNF
from sys import maxsize
from threading import Timer
from sat.base_encoding import BaseEncoding


class DepthAvellaneda(BaseEncoding):
    def __init__(self):
        BaseEncoding.__init__(self)
        self.x = None
        self.f = None
        self.c = None
        self.class_map = None        

    def init_var(self, instance, limit, class_map):
        self.x = {}
        for xl in range(0, len(instance.examples)):
            self.x[xl] = {}
            for x2 in range(0, limit):
                self.x[xl][x2] = self.pool.id(f"x{xl}_{x2}")

        # self.x = [[] for _ in range(0, len(instance.examples))]
        # for xl in self.x:
        #     for _ in range(0, limit):
        #         xl.append(self.add_var())

        self.f = {}
        for i in range(1, 2**limit):
            self.f[i] = {}
            for j in range(1, instance.num_features + 1):
                self.f[i][j] = self.pool.id(f"f{i}_{j}")
        # self.f = [[None] for _ in range(0, 2**limit)]
        # # index starting with 1
        # for i in range(1, 2**limit):
        #     for _ in range(1, instance.num_features + 1):
        #         self.f[i].append(self.add_var())

        c_vars = len(next(iter(class_map.values())))
        self.c = {}
        for i in range(0, 2**limit):
            self.c[i] = {}
            for j in range(0, c_vars):
                self.c[i][j] = self.pool.id(f"c{i}_{j}")

        #self.c = [[self.add_var() for _ in range(0, c_vars)] for _ in range(0, 2**limit)]

    def encode(self, instance, limit):
        self.formula = CNF()
        classes = set()
        for e in instance.examples:
            classes.add(e.cls)
        classes = list(classes) # Give classes an order
        c_vars = len(bin(len(classes)-1)) - 2 # "easier" than log_2

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

        self.init_var(instance, limit, self.class_map)

        # each node has a feature
        for i in range(1, 2**limit):
            clause = []
            for f1 in range(1, instance.num_features + 1):
                clause.append(self.f[i][f1])
                for f2 in range(f1+1, instance.num_features + 1):
                    self.add_clause([-self.f[i][f1], -self.f[i][f2]])            
            self.add_clause(clause)

        for i in range(0, len(instance.examples)):
            self.alg1(instance, i, limit, 0, 1, list())
            self.alg2(instance, i, limit, 0, 1, list(), self.class_map)

        # Forbid non-existing classes
        # Generate all class identifiers
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
                for c_n in range(0, 2**limit):
                    clause = []
                    for i in range(0, c_vars):
                        self.add_clause([self.c[c_n][i] if c_c[i] else -self.c[c_n][i]])
                    self.add_clause(clause)

    def alg1(self, instance, e_idx, limit, lvl, q, clause):
        if lvl == limit:
            return

        example = instance.examples[e_idx]
        for f in range(1, instance.num_features + 1):
            if not example.features[f]:
                self.add_clause([*clause, -self.x[e_idx][lvl], -self.f[q][f]])
        n_cl = list(clause)
        n_cl.append(-self.x[e_idx][lvl])
        self.alg1(instance, e_idx, limit, lvl+1, 2 * q + 1, n_cl)

        for f in range(1, instance.num_features + 1):
            if example.features[f]:
                self.add_clause([*clause, self.x[e_idx][lvl], -self.f[q][f]])
        n_cl2 = list(clause)
        n_cl2.append(self.x[e_idx][lvl])
        self.alg1(instance, e_idx, limit, lvl+1, 2*q, n_cl2)

    def alg2(self, instance, e_idx, limit, lvl, q, clause, class_map):
        if lvl == limit:
            c_vars = class_map[instance.examples[e_idx].cls]
            for i in range(0, len(c_vars)):
                if c_vars[i]:
                    self.add_clause([*clause, self.c[q - 2 ** limit][i]])
                else:
                    self.add_clause([*clause, -self.c[q - 2 ** limit][i]])
        else:
            n_cl = list(clause)
            n_cl.append(self.x[e_idx][lvl])
            n_cl2 = list(clause)
            n_cl2.append(-self.x[e_idx][lvl])
            self.alg2(instance, e_idx, limit, lvl+1, 2*q, n_cl, class_map)
            self.alg2(instance, e_idx, limit, lvl+1, 2*q+1, n_cl2, class_map)

    def run(self, instance, solver, start_bound=1, timeout=0, ub=maxsize):
        c_bound = start_bound
        lb = 0
        best_model = None

        while lb < ub:
            print(f"Running {c_bound}")
            with solver() as slv:
                self.reset_formula()
                try:
                    self.encode(instance, c_bound)
                except MemoryError:
                    return None
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
                    c_bound -= 1
                else:
                    c_bound += 1
                    lb = c_bound

        return best_model

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

        for c_c, c_v in self.class_map.items():
            for i in range(0, num_leafs):
                all_right = True
                for i_v in range(0, len(c_v)):
                    if model[self.c[i][i_v]] != c_v[i_v]:
                        all_right = False
                        break
                if all_right:
                    tree.add_leaf(num_leafs + i, (num_leafs + i)//2, i % 2 == 1, c_c)

        self.reduce_tree(tree, instance)

        if len(next(iter(self.class_map.values()))) > 2:
            return NonBinaryTree(tree)

        return tree

    def reduce_tree(self, tree, instance):
        assigned = {tree.root.id: list(instance.examples)}
        q = [tree.root]
        p = {tree.root.id: None}
        leafs = []

        while q:
            c_n = q.pop()
            examples = assigned[c_n.id]

            if not c_n.is_leaf:
                p[c_n.left.id] = c_n.id
                p[c_n.right.id] = c_n.id
                assigned[c_n.left.id] = []
                assigned[c_n.right.id] = []

                for e in examples:
                    if e.features[c_n.feature]:
                        assigned[c_n.left.id].append(e)
                    else:
                        assigned[c_n.right.id].append(e)

                q.append(c_n.right)
                q.append(c_n.left)
            else:
                leafs.append(c_n)

        for lf in leafs:
            # May already be deleted
            if tree.nodes[lf.id] is None:
                continue

            if len(assigned[lf.id]) == 0:
                c_p = tree.nodes[p[lf.id]]
                o_n = c_p.right if c_p.left.id == lf.id else c_p.left
                if p[c_p.id] is None:
                    tree.root = o_n
                    p[o_n.id] = None
                else:
                    c_pp = tree.nodes[p[c_p.id]]
                    if c_pp.left.id == c_p.id:
                        c_pp.left = o_n
                    else:
                        c_pp.right = o_n
                    p[o_n.id] = c_pp.id

                tree.nodes[lf.id] = None
                tree.nodes[c_p.id] = None

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
