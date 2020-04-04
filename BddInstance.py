class BddExamples:
    def __init__(self, features, cls):
        self.features = features
        self.cls = cls


class BddInstance:
    def __init__(self):
        self.num_features = None
        self.examples = []

    def add_example(self, example):
        if self.num_features is None:
            self.num_features = len(example.features)

        if len(example.features) != self.num_features:
            print(f"Example should have {self.num_features} features, but has {len(example.features)}")
        else:
            # Make 1 based instead of 0 based
            example.features.insert(0, None)
            self.examples.append(example)

