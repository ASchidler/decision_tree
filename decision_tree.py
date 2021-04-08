import os


class DecisionTreeNode:
    def __init__(self, feature, id):
        self.is_leaf = False
        self.feature = feature
        self.left = None
        self.right = None
        self.id = id

    def get_children(self):
        return {True: self.left, False: self.right}

    def __lt__(self, other):
        return self.id < other.id


class DecisionTreeLeaf:
    def __init__(self, cls, id):
        self.is_leaf = True
        self.cls = cls
        self.id = id

    def __lt__(self, other):
        return self.id < other.id


class DecisionTree:
    def __init__(self, num_features, num_nodes):
        self.num_features = num_features
        self.root = None
        self.nodes = [None for _ in range(0, num_nodes + 1)] # Indexing starts at 1

    def copy(self):
        new_tree = DecisionTree(self.num_features, len(self.nodes)-1)

        def deep_copy(cn, parent, polarity):
            if parent is None:
                new_tree.set_root(cn.feature)
            elif cn.is_leaf:
                new_tree.add_leaf(cn.id, parent, polarity, cn.cls)
                return
            else:
                new_tree.add_node(cn.id, parent, cn.feature, polarity)
            deep_copy(cn.left, cn.id, True)
            deep_copy(cn.right, cn.id, False)

        deep_copy(self.root, None, None)
        return new_tree

    def check_consistency(self):
        q = [self.root]
        cnt = 0
        while q:
            e = q.pop()
            cnt += 1

            if not e.is_leaf:
                assert (e.left is not None and e.right is not None)
                q.append(e.left)
                q.append(e.right)
#        assert(cnt == len(self.nodes) - 1)

    def set_root(self, feature):
        if self.root is not None:
            print("Root already set")
        self.root = DecisionTreeNode(feature, 1)
        self.nodes[1] = self.root

    def get_root(self):
        return self.root

    def _add_node(self, id, parent, polarity, node):
        if self.nodes[id] is not None:
            print(f"Node {id} already added")
            raise

        if self.nodes[parent] is None:
            print(f"Parent {parent} not available for node {id}")
            raise

        self.nodes[id] = node
        if polarity:
            if self.nodes[parent].left is not None:
                print("Left child already set")
                raise
            self.nodes[parent].left = node
        else:
            if self.nodes[parent].right is not None:
                print("Right child already set")
                raise
            self.nodes[parent].right = node

        return node

    def add_node(self, id, parent, feature, polarity):
        return self._add_node(id, parent, polarity, DecisionTreeNode(feature, id))

    def add_leaf(self, id, parent, polarity, cls):
        return self._add_node(id, parent, polarity, DecisionTreeLeaf(cls, id))

    def decide(self, features):
        cnode = self.root

        while not cnode.is_leaf:
            if features[cnode.feature]:
                cnode = cnode.left
            else:
                cnode = cnode.right

        return cnode.cls

    def get_path(self, features):
        cnode = self.root

        while not cnode.is_leaf:
            if features[cnode.feature]:
                cnode = cnode.left
            else:
                cnode = cnode.right

        return [cnode] # Leaf is identification enough

    def get_accuracy(self, examples):
        total = 0
        correct = 0
        for e in examples:
            decision = self.decide(e.features)
            total += 1
            if decision == e.cls:
                correct += 1

        return correct / total

    def get_depth(self):
        def dfs_find(node, level):
            if node.is_leaf:
                return level
            else:
                return max(dfs_find(node.left, level + 1), dfs_find(node.right, level + 1))

        return dfs_find(self.root, 0)

    def get_avg_depth(self):
        def dfs_find(node, level):
            if node.is_leaf:
                return level, 1
            else:
                l_d, l_c = dfs_find(node.left, level + 1)
                r_d, r_c = dfs_find(node.right, level + 1)
                return l_d + r_d, l_c + r_c

        t_d, t_c = dfs_find(self.root, 0)
        return t_d / t_c

    def get_nodes(self):
        def dfs_find(node, cnt):
            if node.is_leaf:
                return cnt + 1
            else:
                return dfs_find(node.left, cnt) + dfs_find(node.right, cnt) + 1

        return dfs_find(self.root, 0)

    def clean(self, instance, min_samples=1):
        p = {}
        p[self.root.id] = None
        q = [self.root]

        change = True

        while change:
            change = False

            while q:
                cn = q.pop()
                if cn.left.is_leaf and cn.right.is_leaf:
                    if cn.id not in p or p[cn.id] is None:
                        continue

                    if cn.left.cls == cn.right.cls:
                        if p[cn.id].left == cn:
                            p[cn.id].left = cn.left
                            p[cn.left.id] = p[cn.id]
                        else:
                            p[cn.id].right = cn.left
                            p[cn.left.id] = p[cn.id]
                        self.nodes[cn.right.id] = None
                        self.nodes[cn.id] = None
                        # TODO: This unnecessarily causes a reexploration of the subtree rooted at the parent...
                        # The algorithm can be way more efficient
                        q.append(p[cn.id])
                        p.pop(cn.id)
                else:
                    p[cn.left.id] = cn
                    p[cn.right.id] = cn
                    if not cn.left.is_leaf:
                        q.append(cn.left)
                    if not cn.right.is_leaf:
                        q.append(cn.right)

        leafs = {}
        q = [self.root]
        p = {self.root.id: None}
        while q:
            cn = q.pop()
            if cn.is_leaf:
                leafs[cn.id] = []
            else:
                p[cn.left.id] = cn
                p[cn.right.id] = cn
                q.extend([cn.right, cn.left])

        for e in instance.examples:
            cn = self.root
            while not cn.is_leaf:
                if e.features[cn.feature]:
                    cn = cn.left
                else:
                    cn = cn.right
            leafs[cn.id].append(e)

        for k, v in leafs.items():
            if len(v) < min_samples:
                cp = p[k]

                if p[cp.id] is None:
                    continue

                cpp = p[cp.id]

                if cp.left.id == k:
                    p[cp.right.id] = cpp
                    if cpp.left.id == cp.id:
                        cpp.left = cp.right
                    else:
                        cpp.right = cp.right
                else:
                    p[cp.left.id] = cpp
                    if cpp.left.id == cp.id:
                        cpp.left = cp.left
                    else:
                        cpp.right = cp.left
                change = True
                p.pop(k)
                p.pop(cp.id)
                self.nodes[k] = None
                self.nodes[cp.id] = None

        # Special case root
        while not (self.root.left.is_leaf and self.root.right.is_leaf):
            if self.root.left.is_leaf and len(leafs[self.root.left.id]) == 0:
                self.nodes[self.root.left.id] = None
                self.nodes[self.root.id] = None
                self.root = self.root.right
            elif self.root.right.is_leaf and len(leafs[self.root.right.id]) == 0:
                self.nodes[self.root.right.id] = None
                self.nodes[self.root.id] = None
                self.root = self.root.left
            else:
                break


class NonBinaryNode:
    def __init__(self, id):
        self.is_leaf = False
        self.cls = None
        self.children = {}
        self.feature = None
        self.parent = None
        self.id = id
        self.is_binary = False

    def get_children(self):
        return self.children


class NonBinaryTree:
    def __init__(self, dt=None):
        self.nodes = []
        if dt is None:
            return

        q = [(None, dt.root, None)]
        while q:
            c_p, c_n, c_v = q.pop()
            if not c_n.is_leaf:
                n_n = self.add_node(c_p, c_v, c_n.feature)
                q.append((n_n, c_n.left, True))
                q.append((n_n, c_n.right, False))
            else:
                self.add_leaf(c_p, c_v, c_n.cls)

    def get_root(self):
        return self.nodes[0]

    def _add_node(self, parent, n, value):
        if parent is None:
            if len(self.nodes) > 0:
                print("Double root")
                exit(1)

        if parent is not None:
            if value in self.nodes[parent.id].children:
                print(f"Duplicate nodes for value {value}")
                exit(1)

            self.nodes[parent.id].children[value] = n

        self.nodes.append(n)

    def add_leaf(self, parent, value, cls):
        n_n = NonBinaryNode(len(self.nodes))
        n_n.is_leaf = True
        n_n.cls = cls

        self._add_node(parent, n_n, value)
        return n_n

    def add_node(self, parent, value, feature):
        n_n = NonBinaryNode(len(self.nodes))
        n_n.feature = feature
        self._add_node(parent, n_n, value)
        return n_n

    def decide(self, features):
        cnode = self.nodes[0]

        while not cnode.is_leaf:
            if features[cnode.feature] not in cnode.children:
                return None
            else:
                cnode = cnode.children[features[cnode.feature]]

        return cnode.cls

    def get_path(self, features):
        cnode = self.nodes[0]

        while not cnode.is_leaf:
            while not cnode.is_leaf:
                if features[cnode.feature] not in cnode.children:
                    return [cnode] # The internal node should suffice as identification, as there is only one path there
                else:
                    cnode = cnode.children[features[cnode.feature]]

            if features[cnode.feature]:
                cnode = cnode.left
            else:
                cnode = cnode.right

        return [cnode]  # Leaf is identification enough

    def get_accuracy(self, examples):
        total = 0
        correct = 0
        for e in examples:
            decision = self.decide(e.features)
            total += 1
            if decision == e.cls:
                correct += 1

        return correct / total

    def get_depth(self):
        def dfs_find(node, level):
            if node.is_leaf:
                return level
            else:
                return max(dfs_find(x, level + 1) for x in node.children.values())

        return dfs_find(self.nodes[0], 0)

    def get_nodes(self):
        def dfs_find(node, cnt):
            if node.is_leaf:
                return cnt + 1
            else:
                return sum(dfs_find(x, cnt) for x in node.children.values()) + 1

        return dfs_find(self.nodes[0], 0)

    def check_consistency(self):
        pass


def dot_export(tree):
    output1 = "strict digraph dt {" + os.linesep
    output2 = ""

    q = [tree.root]
    while q:
        cn = q.pop()
        if cn.is_leaf:
            cl = 'red' if cn.cls else 'blue'
            output1 += f"n{cn.id} [label={'1' if cn.cls else '0'}, " \
                       f"shape=box, fontsize=11,width=0.3,height=0.2,fixedsize=true,style=filled,fontcolor=white," \
                       f"color={cl}, fillcolor={cl}];{os.linesep}"
        else:
            output1 += f"n{cn.id} [label={cn.feature},fontsize=11,width=0.4,height=0.25,fixedsize=true];{os.linesep}"
            output2 += f"n{cn.id} -> n{cn.left.id} [color=red, arrowhead=none, len=0.5];{os.linesep}"
            output2 += f"n{cn.id} -> n{cn.right.id} [color=blue, arrowhead=none, len=0.5];{os.linesep}"
            q.append(cn.left)
            q.append(cn.right)

    return output1 + output2 + "}"
