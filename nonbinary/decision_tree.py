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
        self.parent = None
        self.count_left = 0
        self.count_right = 0

    def _decide(self, e):
        # TODO: Find better solution for missing values...
        if e.features[self.feature] == "?":
            if self.count_right > self.count_left:
                return self.right
            else:
                return self.left
        if (self.is_categorical and e.features[self.feature] == self.threshold) \
                or (not self.is_categorical and e.features[self.feature] <= self.threshold):
            return self.left
        else:
            return self.right

    def decide(self, e):
        return self._decide(e).decide(e)

    def get_depth(self):
        return max(self.left.get_depth(), self.right.get_depth()) + 1

    def get_nodes(self):
        return self.left.get_nodes() + self.right.get_nodes() + 1

    def remap(self, mapping):
        if self.feature in mapping:
            self.feature = mapping[self.feature]
        self.left.remap(mapping)
        self.right.remap(mapping)

    def get_children(self):
        return [self.left, self.right]

    def get_path(self, e):
        pth = self._decide(e).get_path(e)
        pth.append(self)
        return pth


class DecisionTreeLeaf:
    def __init__(self, c, i, tree):
        self.is_leaf = True
        self.cls = c
        self.id = i
        self.tree = tree
        self.parent = None

    def decide(self, e):
        return self.cls, self

    def get_depth(self):
        return 0

    def get_nodes(self):
        return 1

    def remap(self, mapping):
        return

    def get_path(self, e):
        return [self]


class DecisionTree:
    def __init__(self):
        self.root = None
        self.nodes = [None]
        self.c_idx = 1

    def set_root_leaf(self, c):
        self.root = DecisionTreeLeaf(c, 1, self)
        self.nodes.append(self.root)
        return self.root

    def set_root(self, f, t, is_categorical=False):
        self.root = DecisionTreeNode(f, t, 1, self, is_categorical)
        self.nodes.append(self.root)
        return self.root

    def _get_id(self):
        while self.c_idx < len(self.nodes) and self.nodes[self.c_idx]:
            self.c_idx += 1
        if self.c_idx >= len(self.nodes):
            self.nodes.append(None)
        return self.c_idx

    def add_node(self, f, t, parent_id, is_left, is_categorical=False):
        c_id = self._get_id()

        new_node = DecisionTreeNode(f, t, c_id, self, is_categorical)
        new_node.parent = self.nodes[parent_id]

        if is_left:
            self.nodes[parent_id].left = new_node
        else:
            self.nodes[parent_id].right = new_node

        self.nodes[c_id] = new_node
        return new_node

    def add_leaf(self, c, parent_id, is_left):
        c_id = self._get_id()
        new_leaf = DecisionTreeLeaf(c, c_id, self)
        new_leaf.parent = self.nodes[parent_id]

        if is_left:
            self.nodes[parent_id].left = new_leaf
        else:
            self.nodes[parent_id].right = new_leaf
        self.nodes[c_id] = new_leaf
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
            n_id = f"c({n.cls})" if n.is_leaf else f"a({n.feature}){' =' if n.is_categorical else ' <='} {n.threshold}"
            lines.append(indent + "" + n_id)
            if not n.is_leaf:
                add_node(n.left, d+1)
                add_node(n.right, d+1)

        add_node(self.root, 0)
        return os.linesep.join(lines)

    def assign(self, instance):
        assigned = defaultdict(list)
        for e in instance.examples:
            pth = self.root.get_path(e)
            for c_node in pth:
                assigned[c_node.id].append(e)
        return assigned

    def clean(self, instance, min_samples=1):
        assigned = self.assign(instance)

        def clean_sub(node):
            if node.is_leaf:
                return
            clean_sub(node.left)
            clean_sub(node.right)

            # Case one, two few examples
            replace_left = node.left.is_leaf and (node.left.id not in assigned or len(assigned[node.left.id]) < min_samples)
            # Case two, both leaves have the same class
            replace_left = replace_left or (node.left.is_leaf and node.right.is_leaf and node.left.cls == node.right.cls)
            replace_right = node.right.is_leaf and (node.right.id not in assigned or len(assigned[node.right.id]) < min_samples)

            if node.parent is not None:
                if replace_left or replace_right:
                    self.c_idx = 1
                    is_left = node.parent.left.id == node.id

                    self.nodes[node.id] = None
                    if replace_left:
                        self.nodes[node.left.id] = None
                        node.right.parent = node.parent
                        assigned[node.right.id] = assigned[node.id]
                        if is_left:
                            node.parent.left = node.right
                        else:
                            node.parent.right = node.right
                    elif replace_right:
                        self.nodes[node.right.id] = None
                        assigned[node.left.id] = assigned[node.id]
                        if is_left:
                            node.parent.left = node.left
                        else:
                            node.parent.right = node.left
            # Special case root
            else:
                if replace_left:
                    self.c_idx = 1
                    self.nodes[node.id] = self.nodes[node.right.id]
                    self.nodes[node.right.id] = None
                    assigned[node.right.id] = assigned[node.id]
                    self.root = node.right
                elif replace_right:
                    self.c_idx = 1
                    self.nodes[node.id] = self.nodes[node.left.id]
                    self.nodes[node.left.id] = None
                    assigned[node.left.id] = assigned[node.id]
                    self.root = node.left

        clean_sub(self.root)
