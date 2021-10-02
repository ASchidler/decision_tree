import os
from collections import defaultdict


class DecisionTreeNode:
    def __init__(self, feature, id):
        self.is_leaf = False
        self.feature = feature
        self.left = None
        self.right = None
        self.id = id

    def get_children(self):
        return {True: self.left, False: self.right}

    def get_leafs(self):
        return self.left.get_leafs() + self.right.get_leafs()

    def __lt__(self, other):
        return self.id < other.id

    def reclassify(self, samples):
        l_samples = []
        r_samples = []
        for c_e in samples:
            if c_e.features[self.feature]:
                l_samples.append(c_e)
            else:
                r_samples.append(c_e)

        self.left.reclassify(l_samples)
        self.right.reclassify(r_samples)


class DecisionTreeLeaf:
    def __init__(self, cls, id):
        self.is_leaf = True
        self.cls = cls
        self.id = id

    def __lt__(self, other):
        return self.id < other.id

    def get_leafs(self):
        return 1

    def reclassify(self, samples):
        c_classes = defaultdict(int)
        for c_e in samples:
            c_classes[c_e.cls] += 1
        if len(c_classes) == 0:
            return
        self.cls = max(c_classes.items(), key=lambda x: x[1])[0]


class DecisionTree:
    def __init__(self, num_features, num_nodes):
        self.num_features = num_features
        self.root = None
        self.nodes = [None for _ in range(0, num_nodes + 1)] # Indexing starts at 1

    def train(self, instance):
        self.root.train(instance)

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
        return self.nodes[1]

    def set_root_leaf(self, cls):
        if self.root is not None:
            print("Root already set")
        self.root = DecisionTreeLeaf(cls, 1)
        self.nodes[1] = self.root
        return self.nodes[1]

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
        """Removes nodes that perform unnecessary splits, i.e. splits where one branch classifies fewer than a minimum number of samples"""
        assigned = self.assign_samples(instance)

        def remove_rec(node):
            if node.is_leaf:
                return None

            if len(assigned[node.left.id]) < min_samples:
                ret = remove_rec(node.right)
                self.nodes[node.left.id] = None
                return node.right if ret is None else ret
            elif len(assigned[node.right.id]) < min_samples:
                ret = remove_rec(node.left)
                self.nodes[node.right.id] = None
                return node.left if ret is None else ret
            else:
                ret1 = remove_rec(node.right)
                if ret1 is not None:
                    self.nodes[node.right.id] = None
                    node.right = ret1
                ret2 = remove_rec(node.left)
                if ret2 is not None:
                    self.nodes[node.left.id] = None
                    node.left = ret2
                return None

        final_root = remove_rec(self.root)
        if final_root is not None:
            self.nodes[self.root.id] = None
            self.root = final_root

    def assign_samples(self, instance):
        assigned_samples = [[] for _ in self.nodes]

        for idx, s in enumerate(instance.examples):
            cnode = self.root
            assigned_samples[cnode.id].append(idx)

            while not cnode.is_leaf:
                if s.features[cnode.feature]:
                    cnode = cnode.left
                else:
                    cnode = cnode.right
                assigned_samples[cnode.id].append(idx)

        return assigned_samples

    def as_string(self):
        lines = []

        def add_node(n, d):
            indent = ''.join("-" for _ in range(0, d))
            n_id = f"c({n.cls})" if n.is_leaf else f"a({n.feature})"
            lines.append(indent + "" + n_id)
            if not n.is_leaf:
                add_node(n.left, d+1)
                add_node(n.right, d+1)

        add_node(self.root, 0)
        return os.linesep.join(lines)


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
