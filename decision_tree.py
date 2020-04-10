class DecisionTreeNode:
    def __init__(self, feature, id):
        self.is_leaf = False
        self.feature = feature
        self.left = None
        self.right = None
        self.id = id


class DecisionTreeLeaf:
    def __init__(self, cls, id):
        self.is_leaf = True
        self.cls = cls
        self.id = id


class DecisionTree:
    def __init__(self, num_features, num_nodes):
        self.num_features = num_features
        self.root = None
        self.nodes = [None for _ in range(0, num_nodes + 1)] # Indexing starts at 1

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

    def add_node(self, id, parent, feature, polarity):
        self._add_node(id, parent, polarity, DecisionTreeNode(feature, id))

    def add_leaf(self, id, parent, polarity, cls):
        self._add_node(id, parent, polarity, DecisionTreeLeaf(cls, id))

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


class DecisionDiagram:
    def __init__(self, num_features, num_nodes):
        self.num_features = num_features
        self.root = None
        self.nodes = [None for _ in range(0, num_nodes + 1)] # Indexing starts at 1
        self.nodes[num_nodes] = DecisionTreeLeaf(False, num_nodes)
        self.nodes[num_nodes-1] = DecisionTreeLeaf(True, num_nodes-1)

    def add_node(self, id, feature, left, right):
        if self.nodes[id] is not None:
            print(f"ERROR: Node {id} already set")

        self.nodes[id] = DecisionTreeNode(feature, id)
        if self.nodes[left] is None:
            print(f"ERROR: For node {id} the left node {left} has not been added")
        else:
            self.nodes[id].left = self.nodes[left]

        if self.nodes[right] is None:
            print(f"ERROR: For node {id} the left node {right} has not been added")
        else:
            self.nodes[id].right = self.nodes[right]

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
        pth = []

        while not cnode.is_leaf:
            pth.append(cnode)
            if features[cnode.feature]:
                cnode = cnode.left
            else:
                cnode = cnode.right

        # TODO: Since the leaf is a conclusion of the preceding path, this can theoretically be skipped...
        pth.append(cnode)
        return pth

    def get_accuracy(self, examples):
        total = 0
        correct = 0
        for e in examples:
            decision = self.decide(e.features)
            total += 1
            if decision == e.cls:
                correct += 1

        return correct / total

    def check_consistency(self):
        pass
