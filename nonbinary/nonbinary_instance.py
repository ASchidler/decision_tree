class Example:
    def __init__(self, inst, features, cls):
        self.cls = cls
        self.features = [None, *features]
        self.instance = inst
        self.id = None


class ClassificationInstance:
    def __init__(self):
        self.examples = []
        self.domains = []
        self.num_features = -1
        self.classes = set()
        self.domains = []
        self.is_binary = set()
        self.is_categorical = set()
        self.feature_idx = dict()
        self.feature_indices = -1

    def finish(self):
        c_idx = 1
        for i in range(1, self.num_features+1):
            if len(self.domains[i]) <= 2:
                self.is_binary.add(i)
            if any(isinstance(x, str) and x != "?" for x in self.domains[i]):
                self.is_categorical.add(i)
            self.feature_idx[i] = c_idx
            c_idx += len(self.domains[i])
            self.domains[i] = sorted(list(self.domains[i]))
        self.feature_indices = c_idx - 1

    def add_example(self, e):
        if self.num_features == -1:
            self.num_features = len(e.features) - 1
            self.domains = [set() for _ in range(0, self.num_features + 1)]
        elif len(e.features) - 1 != self.num_features:
            raise RuntimeError("Examples have different number of features")

        e.id = len(self.examples)
        self.examples.append(e)
        for i in range(1, self.num_features+1):
            if e.features[i] != "?":
                self.domains[i].add(e.features[i])
        self.classes.add(e.cls)


def parse(filename):
    instance = ClassificationInstance()

    with open(filename, "r") as f:
        for ln in f:
            fields = ln.split(',')
            example = []

            for i, fd in enumerate(fields[:-1]):
                # Not binary values, skip line
                fd = fd.strip()

                try:
                    fd = int(fd)
                except ValueError:
                    try:
                        fd = float(fd)
                    except ValueError:
                        pass
                example.append(fd)

            cls = fields[-1].strip()
            instance.add_example(Example(instance, example, cls))

    instance.finish()
    return instance

