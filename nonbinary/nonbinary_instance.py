import os
import random


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

    def _verify_support_set(self, supset):
        for i in range(0, len(self.examples)):
            for j in range(i+1, len(self.examples)):
                if self.examples[i].cls != self.examples[j].cls:
                    found = False
                    for c_f in range(1, self.num_features + 1):
                        if self.examples[i][c_f] != self.examples[j][c_f] and c_f in supset:
                            found = True
                            break
                    if not found:
                        raise RuntimeError("Not a real support set.")

    def min_key_random(self, randomize=True):
        supset = set()
        features = list(range(1, self.num_features + 1))

        if randomize:
            random.shuffle(features)
        for i in range(0, len(self.examples)):
            for j in range(i+1, len(self.examples)):
                if self.examples[i].cls != self.examples[j].cls:
                    found = False
                    for c_f in supset:
                        if self.examples[i][c_f] != self.examples[j][c_f]:
                            found = True
                    if not found:
                        for c_f in reversed(features)

    def min_key_removal(self, randomize=True):
        pass

    def min_key_greedy(self):

def parse(path, filename, slice, use_validation=False, use_test=True):
    target_files = [x for x in os.listdir(path) if x.startswith(filename + ".") and x.endswith(".data")]
    if len(target_files) == 0:
        raise RuntimeError("No data files found with this name.")

    target_files.sort()

    test_file = None
    if os.path.exists(os.path.join(path, filename + ".test")):
        if slice != 1:
            raise RuntimeError("File has an existing test set, slice cannot be different than 1")
        test_file = _parse_file([os.path.join(path, filename + ".test")])
    elif use_test:
        target_idx = (slice + 3) % len(target_files)
        test_file = _parse_file([os.path.join(path, target_files[target_idx])])
        target_files.pop(target_idx)

    validation_file = None
    if use_validation:
        target_idx = (slice + 2) % len(target_files)
        validation_file = _parse_file([os.path.join(path, target_files[target_idx])])
        target_files.pop(target_idx)

    data_file = _parse_file([os.path.join(path, x) for x in target_files])

    return data_file, test_file, validation_file


def _parse_file(filenames):
    instance = ClassificationInstance()

    for filename in filenames:
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

