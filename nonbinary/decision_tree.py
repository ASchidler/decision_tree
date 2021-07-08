import os
from collections import defaultdict


class DecisionTreeNode:
    def __init__(self, f, t, i, tree, is_categorical=False):
        self.is_leaf = False
        self.threshold = t
        self.feature = f
        self.left = None
        self.right = None
        self.id = i
        self.tree = tree
        self.is_categorical = is_categorical

    def decide(self, e):
        if (not self.is_categorical and e.features[self.feature] <= self.threshold) or \
            (self.is_categorical and e.features[self.feature] == self.threshold):
            return self.left.decide(e)
        else:
            return self.right.decide(e)

    def get_depth(self):
        return max(self.left.get_depth(), self.right.get_depth()) + 1

    def get_nodes(self):
        return self.left.get_nodes() + self.right.get_nodes() + 1


class DecisionTreeLeaf:
    def __init__(self, c, i, tree):
        self.is_leaf = True
        self.cls = c
        self.id = i
        self.tree = tree

    def decide(self, e):
        return self.cls, self

    def get_depth(self):
        return 0

    def get_nodes(self):
        return 1


class DecisionTree:
    def __init__(self):
        self.root = None
        self.nodes = [None]

    def set_root_leaf(self, c):
        self.root = DecisionTreeLeaf(c, 1, self)
        self.nodes.append(self.root)
        return self.root

    def set_root(self, f, t, is_categorical=False):
        self.root = DecisionTreeNode(f, t, 1, self, is_categorical)
        self.nodes.append(self.root)
        return self.root

    def add_node(self, f, t, parent, is_left, is_categorical=False):
        new_node = DecisionTreeNode(f, t, len(self.nodes), self, is_categorical)
        if is_left:
            self.nodes[parent].left = new_node
        else:
            self.nodes[parent].right = new_node
        self.nodes.append(new_node)
        return new_node

    def add_leaf(self, c, parent, is_left):
        new_leaf = DecisionTreeLeaf(c, len(self.nodes), self)
        if is_left:
            self.nodes[parent].left = new_leaf
        else:
            self.nodes[parent].right = new_leaf
        self.nodes.append(new_leaf)
        return new_leaf

    def get_accuracy(self, examples):
        errors = 0
        for c_e in examples:
            if self.root.decide(c_e)[0] != c_e.cls:
                errors += 1

        return (len(examples) - errors) / len(examples)

    def get_depth(self):
        return self.root.get_depth()

    def get_nodes(self):
        return self.root.get_nodes()

    def as_string(self):
        lines = []

        def add_node(n, d):
            indent = ''.join("-" for _ in range(0, d))
            n_id = f"c({n.cls})" if n.is_leaf else f"a({n.feature}): {n.threshold}"
            lines.append(indent + "" + n_id)
            if not n.is_leaf:
                add_node(n.left, d+1)
                add_node(n.right, d+1)

        add_node(self.root, 0)
        return os.linesep.join(lines)

    def assign(self, examples):
        assigned = defaultdict(list)
        for e in examples:
            _, node = self.root.decide(e)
            assigned[node.id].append(e)
        return assigned

    def clean(self, examples, min_samples=1):
        assigned = self.assign(examples)

        def clean_sub(node):
            if node.is_leaf:
                if node.id not in assigned or len(assigned[node.id]) < min_samples:
                    self.nodes[node.id] = None
                    return True, None
                return None
            else:
                result_l = clean_sub(node.left)
                result_r = clean_sub(node.right)

                if result_l:
                    if result_l[1]:
                        node.left = result_l[1]
                    else:
                        return True, node.right
                elif result_r:
                    if result_r[1]:
                        node.right = result_r[1]
                    else:
                        return True, node.left

                if node.left.is_leaf and node.right.is_leaf and node.right.cls == node.left.cls:
                    return True, node.left

        clean_sub(self.root)
