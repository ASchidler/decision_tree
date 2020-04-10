class BddExamples:
    def __init__(self, features, cls):
        self.features = features
        self.cls = cls


class BddInstance:
    def __init__(self):
        self.num_features = None
        self.examples = []
        self.reduce_map = None

    def add_example(self, example):
        if self.num_features is None:
            self.num_features = len(example.features)

        if example.features[0] is not None:
            if len(example.features) != self.num_features:
                print(f"Example should have {self.num_features} features, but has {len(example.features)}")
            else:
                # Make 1 based instead of 0 based
                example.features.insert(0, None)
                self.examples.append(example)
        else:
            if len(example.features) != self.num_features + 1:
                print(f"Example should have {self.num_features} features, but has {len(example.features) - 1}")
            else:
                self.examples.append(example)

    def reduce_same_features(self):
        unnecessary = []

        for i in range(1, self.num_features + 1):
            cVal = None
            found = False
            for e in self.examples:
                if cVal is None:
                    cVal = e.features[i]

                if e.features[i] != cVal:
                    found = True
                    break

            if not found:
                unnecessary.append(i)

        cIdx = self.num_features
        self.reduce_map = {}
        for u in unnecessary:
            while cIdx in unnecessary:
                cIdx -= 1

            if cIdx <= 1:
                break

            self.reduce_map[u] = cIdx
            for e in self.examples:
                e.features[u], e.features[cIdx] = e.features[cIdx], e.features[u]
            cIdx -= 1

        self.num_features -= len(self.reduce_map)
        print(f"Reduced {len(self.reduce_map)} features")

    def unreduce_instance(self, tree):
        for v1, v2 in self.reduce_map.items():
            for n in tree.nodes:
                if n is not None and not n.is_leaf:
                    if n.feature == v1:
                        n.feature = v2
            for e in self.examples:
                e.features[v1], e.features[v2] = e.features[v2], e.features[v1]
        self.num_features += len(self.reduce_map)

