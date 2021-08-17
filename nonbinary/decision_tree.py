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

    def train(self, samples):
        ls = []
        rs = []

        for e in samples:
            target = self._decide(e)
            if target.id == self.left.id:
                ls.append(e)
            else:
                rs.append(e)
        self.count_left = len(ls)
        self.count_right = len(rs)
        self.left.train(ls)
        self.right.train(rs)

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

    def get_leaves(self):
        return self.left.get_leaves() + self.right.get_leaves()

    def get_extended_leaves(self):
        return self.left.get_extended_leaves() + self.right.get_extended_leaves()

    def reclassify(self, samples):
        ls = []
        rs = []

        for e in samples:
            target = self._decide(e)
            if target.id == self.left.id:
                ls.append(e)
            else:
                rs.append(e)

        self.left.reclassify(ls)
        self.right.reclassify(rs)

    def copy(self, new_tree, c_p, is_left):
        if c_p is None:
            n_n = new_tree.set_root(self.feature, self.threshold, self.is_categorical)
        else:
            n_n = new_tree.add_node(self.feature, self.threshold, c_p.id, is_left, self.is_categorical)
        self.left.copy(new_tree, n_n, True)
        self.right.copy(new_tree, n_n, False)


class DecisionTreeLeaf:
    def __init__(self, c, i, tree):
        self.is_leaf = True
        self.cls = c
        self.id = i
        self.tree = tree
        self.parent = None
        self.count = 0

    def decide(self, e):
        return self.cls, self

    def train(self, samples):
        self.count = len(samples)

    def get_depth(self):
        return 0

    def get_nodes(self):
        return 1

    def get_leaves(self):
        return 1

    def remap(self, mapping):
        return

    def get_path(self, e):
        return [self]

    def get_extended_leaves(self):
        return 0 if self.cls.startswith("-") or self.cls == "EmptyLeaf" else 1

    def reclassify(self, samples):
        classes = defaultdict(int)
        for cs in samples:
            classes[cs.cls] += 1

        _, self.cls = max((v, k) for k, v in classes.items())

    def copy(self, new_tree, c_p, is_left):
        if c_p is None:
            new_tree.set_root_leaf(self.cls)
        else:
            new_tree.add_leaf(self.cls, c_p.id, is_left)

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

    def train(self, instance):
        self.root.train(instance.examples)

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
            decision = self.root.decide(c_e)[0]
            if decision != c_e.cls and (c_e.surrogate_cls is None or c_e.surrogate_cls != decision):
                errors += 1

        return (len(examples) - errors) / len(examples)

    def get_depth(self):
        return self.root.get_depth()

    def get_nodes(self):
        return self.root.get_nodes()

    def as_string(self):
        lines = []

        def add_node(n, indent):
            n_id = f"c({n.cls})" if n.is_leaf else \
                f"a({n.feature}){' =' if n.is_categorical else ' <='} {n.threshold}"
            lines.append(indent + "" + n_id)
            if not n.is_leaf:
                add_node(n.left, indent+"-")
                add_node(n.right, indent+"-")

        add_node(self.root, "")
        return os.linesep.join(lines)

    def assign(self, instance):
        assigned = defaultdict(list)
        for e in instance.examples:
            pth = self.root.get_path(e)
            for c_node in pth:
                assigned[c_node.id].append(e)
        return assigned

    def copy(self):
        n_tree = DecisionTree()
        self.root.copy(n_tree, None, None)
        return n_tree

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
            replace_left = replace_left or (node.left.is_leaf and node.left.cls == "EmptyLeaf")
            replace_right = node.right.is_leaf and (node.right.id not in assigned or len(assigned[node.right.id]) < min_samples)
            replace_right = replace_right or (node.right.is_leaf and node.right.cls == "EmptyLeaf")

            if replace_left or replace_right:
                if node.parent is not None:
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
                        node.left.parent = node.parent
                        assigned[node.left.id] = assigned[node.id]
                        if is_left:
                            node.parent.left = node.left
                        else:
                            node.parent.right = node.left
                # Special case root
                else:
                    self.c_idx = 1
                    if replace_left:
                        self.nodes[node.id] = self.nodes[node.right.id]
                        self.nodes[node.right.id] = None
                        self.nodes[node.left.id] = None
                        node.right.id = node.id
                        node.right.parent = None
                        assigned[node.right.id] = assigned[node.id]
                        self.root = node.right
                    elif replace_right:
                        self.nodes[node.id] = self.nodes[node.left.id]
                        self.nodes[node.left.id] = None
                        self.nodes[node.right.id] = None
                        node.left.id = node.id
                        node.left.parent = None
                        assigned[node.left.id] = assigned[node.id]
                        self.root = node.left

        clean_sub(self.root)

    def check(self):
        """Perform a sanity check."""
        q = [self.root]
        nodes = set()

        while q:
            c_n = q.pop()
            if c_n.id in nodes:
                print("Duplicate node id or loop")
            nodes.add(c_n.id)

            if self.nodes[c_n.id] != c_n:
                print("Node id doesn't match node")

            if not c_n.is_leaf:
                q.extend(c_n.get_children())
                if c_n.left is None or c_n.right is None:
                    print("Child is none")

            if c_n.parent is not None and not (c_n.parent.left.id == c_n.id or c_n.parent.right.id == c_n.id):
                print("Invalid parent/child relationship")

            if c_n.parent is None and c_n != self.root:
                print("Double root")

        if any(x.id not in nodes for x in self.nodes if x is not None):
            print("Superfluous nodes")


def tree_compare(n1, n2):
    if n1.is_leaf != n2.is_leaf:
        raise RuntimeError("Leaf non leaf.")

    if not n1.is_leaf:
        if n1.feature != n2.feature:
            raise RuntimeError("Features diverge.")
        if n1.threshold != n2.threshold:
            raise RuntimeError("Threshold diverge.")
        if n1.is_categorical != n2.is_categorical:
            raise RuntimeError("Categorical diverge.")
        tree_compare(n1.left, n2.left)
        tree_compare(n1.right, n2.right)
    elif n1.cls != n2.cls:
        raise RuntimeError("Divergent classes.")
