import os
import random
from decimal import Decimal, InvalidOperation

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
        self.reduced_key = None
        self.reduced_map = None
        self.reduced_original_num_features = None
        self.reduced_dropped = None

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

    def min_key_random(self, randomize=False):
        supset = set()
        features = list(range(1, self.num_features + 1))

        if randomize:
            random.shuffle(features)
        else:
            features.sort(key=lambda x: len(self.domains[x]))

        for i in range(0, len(self.examples)):
            for j in range(i+1, len(self.examples)):
                if self.examples[i].cls != self.examples[j].cls:
                    found = False

                    # Check if there is a disagreement for any feature in the set
                    for c_f in supset:
                        if self.examples[i].features[c_f] != self.examples[j].features[c_f]:
                            found = True
                            break

                    if not found:
                        for c_fi, c_f in enumerate(features):
                            if self.examples[i].features[c_f] != self.examples[j].features[c_f]:
                                supset.add(c_f)
                                features[c_fi], features[-1] = features[-1], features[c_fi]
                                features.pop()

        return supset

    def min_key_removal(self, randomize=False):
        supset = set(range(1, self.num_features+1))
        features = list(range(1, self.num_features+1))

        if randomize:
            random.shuffle(features)
        else:
            features.sort(key=lambda x: -len(self.domains[x]))

        for c_f in features:
            violated = False
            supset.remove(c_f)

            for i in range(0, len(self.examples)):
                if violated:
                    break
                for j in range(i + 1, len(self.examples)):
                    if self.examples[i].cls != self.examples[j].cls:
                        found = False
                        for c_f2 in supset:
                            # Check if there is a disagreement for any feature in the set
                            if self.examples[i].features[c_f2] != self.examples[j].features[c_f2]:
                                found = True
                                break
                        if not found:
                            violated = True
                            break
            if violated:
                supset.add(c_f)

        return supset

    def min_key_greedy(self):
        violated = True
        supset = set()
        non_supset = set(range(1, self.num_features + 1))

        while violated:
            violated = False
            counts = [0 for _ in range(0, self.num_features + 1)]
            for i in range(0, len(self.examples)):
                for j in range(i + 1, len(self.examples)):
                    if self.examples[i].cls != self.examples[j].cls:
                        found = False
                        for c_f in supset:
                            if self.examples[i].features[c_f] != self.examples[j].features[c_f]:
                                found = True
                                break
                        # Check with features the two lines would disagree on
                        if not found:
                            violated = True
                            for c_f in non_supset:
                                if self.examples[i].features[c_f] != self.examples[j].features[c_f]:
                                    counts[c_f] += 1

            if violated:
                _, _, n_f = max((counts[i], -len(self.domains[i]), i) for i in non_supset)
                non_supset.remove(n_f)
                supset.add(n_f)

        return supset

    def test_key(self, key):
        for i in range(0, len(self.examples)):
            for j in range(i + 1, len(self.examples)):
                if self.examples[i].cls != self.examples[j].cls:
                    found = False
                    for c_f in key:
                        if self.examples[i].features[c_f] != self.examples[j].features[c_f]:
                            found = True
                            break
                    if not found:
                        return False
        return True

    def reduce(self, key):
        if len(key) == self.num_features:
            return

        if self.reduced_key is not None:
            raise RuntimeError("Instance has already been reduced")

        self.reduced_map = dict()
        self.reduced_key = key
        self.reduced_original_num_features = self.num_features

        # Map features such that all features in the key are front
        c_tail = self.num_features
        c_head = 1
        while c_head < c_tail:
            if c_head in key:
                # self.reduced_map[c_head] = c_head
                c_head += 1
            elif c_tail not in key:
                # self.reduced_map[c_tail] = c_tail
                c_tail -= 1
            else:
                self.reduced_map[c_tail] = c_head
                c_head += 1
                c_tail -= 1

        # Swap features
        for c_e in self.examples:
            for c_k, c_v in self.reduced_map.items():
                c_e.features[c_k], c_e.features[c_v] = c_e.features[c_v], c_e.features[c_k]

        self.num_features = len(key)

        # Eliminate duplicates
        known_entries = dict()
        self.reduced_dropped = []
        for c_e in self.examples:
            values = tuple(c_e.features[1:self.num_features+1])

            if values in known_entries:
                if known_entries[values] != c_e.cls:
                    raise RuntimeError("Key is not a real key, duplicate with different classes found.")
                self.reduced_dropped.append(c_e)
            else:
                known_entries[values] = c_e.cls

        for c_e in reversed(self.reduced_dropped):
            self.examples[c_e.id], self.examples[-1] = self.examples[-1], self.examples[c_e.id]
            self.examples.pop()

        # Update ids
        for i, c_e in enumerate(self.examples):
            c_e.id = i

    def unreduce(self, decision_tree=None):
        if self.reduced_map is None:
            return

        self.num_features = self.reduced_original_num_features

        for c_e in self.reduced_dropped:
            c_e.id = len(self.examples)
            self.examples.append(c_e)

        self.reduced_dropped = None

        reverse_lookup = {x: y for y, x in self.reduced_map.items()}
        if decision_tree:
            decision_tree.root.remap(reverse_lookup)

        for c_e in self.examples:
            for c_k, c_v in reverse_lookup.items():
                c_e.features[c_k], c_e.features[c_v] = c_e.features[c_v], c_e.features[c_k]
        self.reduced_map = None

        self.reduced_original_num_features = None
        self.reduced_key = None


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
                    if fd == "na":
                        fd = "?"

                    try:
                        fd = int(fd)
                    except ValueError:
                        try:
                            fd = Decimal(fd)
                        except ValueError:
                            pass
                        except InvalidOperation:
                            pass

                    example.append(fd)


                cls = fields[-1].strip()
                instance.add_example(Example(instance, example, cls))

    instance.finish()
    return instance

