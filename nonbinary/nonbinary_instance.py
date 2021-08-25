import os
import random
from decimal import Decimal, InvalidOperation, getcontext
from collections import defaultdict


class Example:
    def __init__(self, inst, features, cls, surrogate_cls=None):
        self.cls = cls
        self.features = [None, *features]
        self.instance = inst
        self.id = None
        self.surrogate_cls = surrogate_cls
        self.original_values = None

    def copy(self, instance):
        return Example(instance, self.features[1:], self.cls, self.surrogate_cls)


class ClassificationInstance:
    def __init__(self):
        self.examples = []
        self.domains = []
        self.num_features = -1
        self.classes = set()
        self.has_missing = False
        self.domains = []
        self.domain_counts = []
        self.domains_max = []
        self.is_binary = set()
        self.is_categorical = set()
        self.feature_idx = dict()
        self.feature_indices = -1
        self.reduced_key = None
        self.reduced_map = None
        self.reduced_original_num_features = None
        self.reduced_dropped = None
        self.class_sizes = None
        self.layer_reduced = False

    def finish(self):
        c_idx = 1
        self.domains_max = [0 for _ in range(0, self.num_features + 1)]

        # TODO: use average for non-categorial values...

        for i in range(1, self.num_features+1):
            if len(self.domains[i]) <= 2:
                self.is_binary.add(i)
            if any(isinstance(x, str) and x != "?" for x in self.domains[i]):
                self.is_categorical.add(i)
            self.feature_idx[i] = c_idx
            c_idx += len(self.domains[i])
            self.domains[i] = sorted(list(self.domains[i]))
            if len(self.domains[i]) > 0:
                if i in self.is_categorical:
                    _, self.domains_max[i] = max((v, k) for k, v in self.domain_counts[i].items())
                else:
                    c_sum = 0
                    c_cnt = 0
                    is_int = True
                    for k, v in self.domain_counts[i].items():
                        c_sum += k * v
                        c_cnt += v
                        is_int = is_int and isinstance(k, int)
                    #self.domains_max[i] = c_sum // c_cnt if is_int else c_sum / c_cnt
                    self.domains_max[i] =  c_sum / c_cnt

        self.feature_indices = c_idx - 1

    def add_example(self, e):
        if self.num_features == -1:
            self.num_features = len(e.features) - 1
            self.domains = [set() for _ in range(0, self.num_features + 1)]
            self.domain_counts = [defaultdict(int) for _ in range(0, self.num_features + 1)]
        elif len(e.features) - 1 != self.num_features:
            raise RuntimeError("Examples have different number of features")

        e.id = len(self.examples)
        self.examples.append(e)
        for i in range(1, self.num_features+1):
            if e.features[i] != "?":
                self.domains[i].add(e.features[i])
                self.domain_counts[i][e.features[i]] += 1
            else:
                self.has_missing = True
        self.classes.add(e.cls)
        if e.surrogate_cls:
            self.classes.add(e.surrogate_cls)

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

    def min_key_random(self):
        supset = []
        features = list(range(1, self.num_features + 1))
        random.shuffle(features)

        # Group examples by classes, if the partitions are non-trivially small, this significantly improves runtime
        classes = defaultdict(list)
        for c_e in self.examples:
            classes[c_e.cls].append(c_e)
        iterations = sum(len(v1) * len(v2) for (k1, v1) in classes.items() for (k2, v2) in classes.items() if k1 < k2)

        # Check for each pair of examples
        for c_c, c_es in classes.items():
            for c_c2, c_es2 in classes.items():
                if c_c >= c_c2:
                    continue

                for c_e1 in c_es:
                    for c_e2 in c_es2:
                        found = False

                        # Check if there is a disagreement for any feature in the set
                        for c_f, c_v in supset:
                            if c_e1.features[c_f] == "?" or c_e2.features[c_f] == "?":
                                continue

                            if c_f in self.is_categorical:
                                if (c_e1.features[c_f] == c_v) ^ (c_e2.features[c_f] == c_v):
                                    found = True
                                    break
                            elif (c_e1.features[c_f] <= c_v < c_e2.features[c_f])\
                                    or (c_e1.features[c_f] > c_v >= c_e2.features[c_f]):
                                found = True
                                break

                        if not found:
                            for c_f in features:
                                if c_e1.features[c_f] == "?" or c_e2.features[c_f] == "?":
                                    continue

                                if c_e1.features[c_f] != c_e2.features[c_f]:
                                    if c_f in self.is_categorical:
                                        supset.append((c_f, c_e1.features[c_f]))
                                    else:
                                        supset.append((c_f, min(c_e1.features[c_f], c_e2.features[c_f])))
                                    break
                            random.shuffle(features)

        return supset

    def reduce_with_key(self, randomized_runs=1, only_features=False):
        keys = []
        for _ in range(0, randomized_runs):
            keys.append(self.min_key_random())

        key = min(keys, key=lambda x: len(x))
        if only_features:
            key = set(x[0] for x in key)
        self.reduce(key)

    def test_key(self, key):
        for i in range(0, len(self.examples)):
            for j in range(i + 1, len(self.examples)):
                if self.examples[i].cls != self.examples[j].cls:
                    found = False
                    for c_f, c_v in key:
                        if c_f in self.is_categorical:
                            if (self.examples[i].features[c_f] == c_v) ^ (self.examples[j].features[c_f] == c_v):
                                found = True
                                break
                        elif (self.examples[i].features[c_f] <= c_v < self.examples[j].features[c_f] or
                                self.examples[i].features[c_f] > c_v >= self.examples[j].features[c_f]):
                            found = True
                            break
                    if not found:
                        return False
        return True

    def reduce(self, key):
        # Nothing to reduce
        if len(key) == self.num_features:
            return

        # Trivial decision tree, as a leaf will suffice
        if len(key) == 0:
            return

        if self.reduced_key is not None:
            raise RuntimeError("Instance has already been reduced")

        reduce_features = set()
        reduce_thresholds = defaultdict(list)
        if isinstance(next(iter(key)), int):
            reduce_features = key
        else:
            for c_f, c_v in key:
                if len(self.domains[c_f]) > 0:
                    reduce_features.add(c_f)
                    reduce_thresholds[c_f].append(c_v)

            for c_l in reduce_thresholds.values():
                c_l.sort()

            for c_f in reduce_features:
                if c_f not in self.is_categorical:
                    reduce_thresholds[c_f].append(self.domains[c_f][-1])
                else:
                    reduce_thresholds[c_f] = set(reduce_thresholds[c_f])

        self.reduced_map = dict()
        self.reduced_key = reduce_features
        self.reduced_original_num_features = self.num_features

        # Map features such that all features in the key are front
        c_tail = self.num_features
        c_head = 1
        while c_head < c_tail:
            if c_head in reduce_features:
                # self.reduced_map[c_head] = c_head
                c_head += 1
            elif c_tail not in reduce_features:
                # self.reduced_map[c_tail] = c_tail
                c_tail -= 1
            else:
                self.reduced_map[c_tail] = c_head
                c_head += 1
                c_tail -= 1

        # Map values
        if len(reduce_thresholds) > 0:
            for c_e in self.examples:
                c_e.original_values = list(c_e.features)
                for c_f in reduce_features:
                    if c_e.features[c_f] == "?":
                        continue

                    if c_f in self.is_categorical:
                        if c_e.features[c_f] not in reduce_thresholds[c_f]:
                            c_e.features[c_f] = "DummyValue"
                    else:
                        for c_v in reduce_thresholds[c_f]:
                            if c_e.features[c_f] <= c_v:
                                c_e.features[c_f] = c_v
                                break

            for c_f, c_v in reduce_thresholds.items():
                self.domains[c_f] = c_v

        # Swap features
        for c_e in self.examples:
            for c_k, c_v in self.reduced_map.items():
                c_e.features[c_k], c_e.features[c_v] = c_e.features[c_v], c_e.features[c_k]
        for c_k, c_v in self.reduced_map.items():
            self.domains[c_k], self.domains[c_v] = self.domains[c_v], self.domains[c_k]
            self.domain_counts[c_k], self.domain_counts[c_v] = self.domain_counts[c_v], self.domain_counts[c_k]
            self.domains_max[c_k], self.domains_max[c_v] = self.domains_max[c_v], self.domains_max[c_k]
        self.finish()

        self.num_features = len(reduce_features)

        # Eliminate duplicates
        known_entries = dict()
        self.reduced_dropped = []
        for c_e in self.examples:
            values = tuple(c_e.features[1:self.num_features+1])

            if values in known_entries:
                if known_entries[values] != c_e.cls:
                    print(f"Not real key error {values}")
                    #raise RuntimeError("Key is not a real key, duplicate with different classes found.")
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
            if c_e.original_values is not None:
                c_e.features = c_e.original_values
                c_e.original_values = None
                #TODO: Fix domains...
            else:
                for c_k, c_v in reverse_lookup.items():
                    c_e.features[c_k], c_e.features[c_v] = c_e.features[c_v], c_e.features[c_k]
        for c_k, c_v in reverse_lookup.items():
            self.domains[c_k], self.domains[c_v] = self.domains[c_v], self.domains[c_k]
            self.feature_idx[c_k], self.feature_idx[c_v] = self.feature_idx[c_v], self.feature_idx[c_k]
            self.domain_counts[c_k], self.domain_counts[c_v] = self.domain_counts[c_v], self.domain_counts[c_k]
            self.domains_max[c_k], self.domains_max[c_v] = self.domains_max[c_v], self.domains_max[c_k]
        self.reduced_map = None

        self.reduced_original_num_features = None
        self.reduced_key = None

        self.finish()

    def export_c45(self, path, write_names=True):
        with open(path, "w") as outp:
            for c_e in self.examples:
                for c_f in c_e.features[1:]:
                    outp.write(f"{c_f},")
                outp.write(f"{c_e.cls}{os.linesep}")
        if write_names:
            with open(path[:-4] + "names", "w") as outp:
                # Class
                for c_i, c_v in enumerate(self.classes):
                    if c_i != 0:
                        outp.write(",")
                    outp.write(f"{c_v}")
                outp.write("." + os.linesep)

                for c_i, c_d in enumerate(self.domains[1:]):
                    is_cat = any(isinstance(x, str) for x in c_d)

                    outp.write(f"att{c_i + 1}: ")
                    if not is_cat:
                        outp.write("continuous." + os.linesep)
                    else:
                        for c_di, c_v in enumerate(c_d):
                            if c_di != 0:
                                outp.write(",")
                            outp.write(f"{c_v}")
                        outp.write("." + os.linesep)


def parse(path, filename, slice, use_validation=False, use_test=True):
    target_files = [x for x in os.listdir(path) if x.startswith(filename + ".") and x.endswith(".data")]
    if len(target_files) == 0:
        raise RuntimeError("No data files found with this name.")

    target_files.sort()

    test_file = None
    if os.path.exists(os.path.join(path, filename + ".test")):
        if slice != 1:
            raise FileNotFoundError("File has an existing test set, slice cannot be different than 1")
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
    getcontext().prec = 6

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
                            decimals = len(fd.split(".")[-1])
                            if decimals <= 6:
                                fd = Decimal(fd)
                            else:
                                # Truncate after 6 decimals, since weka does this
                                Decimal(fd)
                                fd = Decimal(fd[:-(decimals-6)])
                        except InvalidOperation:
                            try:
                                # Special case to catch scientific notation
                                fd = float(fd)
                                # Round to 6 decimals
                                fd = Decimal(int(fd * 1000000)) / Decimal(1000000.0)
                            except ValueError:
                                pass

                    example.append(fd)


                cls = fields[-1].strip()
                instance.add_example(Example(instance, example, cls))

    instance.finish()
    return instance

