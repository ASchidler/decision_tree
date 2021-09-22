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
        self.impurities = None

    def copy(self, instance):
        return Example(instance, self.features[1:], self.cls, self.surrogate_cls)


class ClassificationInstance:
    def __init__(self):
        self.examples = []
        self.domains = []
        self.num_features = -1
        self.classes = set()
        self.class_distribution = defaultdict(int)
        self.has_missing = False
        self.domains = []
        self.domain_counts = []
        self.domains_max = []
        self.is_categorical = set()
        self.feature_idx = dict()
        self.reduced_key = None
        self.reduced_map = None
        self.reduced_original_num_features = None
        self.reduced_dropped = None
        self.reduced_domains = None
        self.reduced_domains_max = None
        self.reduced_domain_counts = None
        self.reduced_feature_idx = None
        self.reduced_categorical = None

        self.class_sizes = None
        self.layer_reduced = False

    def finish(self, base_instance=None, clean_domains=True):
        c_idx = 1
        self.domains_max = [0 for _ in range(0, self.num_features + 1)]

        for i in range(1, self.num_features+1):
            if any(isinstance(x, str) and x != "?" for x in self.domains[i]):
                self.is_categorical.add(i)

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
                    self.domains_max[i] = c_sum / c_cnt

            if clean_domains and i not in self.is_categorical:
                new_domain = []
                values = [(e.features[i], e.cls) for e in self.examples]
                values.sort()

                c_cls = None
                last_val = None
                last_last_val = None

                for c_v, c_c in values:
                    if c_cls is None:
                        c_cls = c_c
                    else:
                        if c_cls != c_c:
                            c_cls = c_c
                            if last_val == c_v:  # Both classes with the same value
                                if last_last_val is not None and (not new_domain or new_domain[-1] != last_last_val):
                                    new_domain.append(last_last_val)

                            if last_val is not None and (not new_domain or new_domain[-1] != last_val):
                                new_domain.append(last_val)

                    if c_v != last_val:
                        last_last_val = last_val
                        last_val = c_v

                # Add last value so the max is in the domain, will be ignored by the encodings
                if self.domains[i]:
                    maximum = max(self.domains[i])
                    if not new_domain or new_domain[-1] != maximum:
                        new_domain.append(maximum)
                self.domains[i] = new_domain
            else:
                self.domains[i] = sorted(list(self.domains[i]))

            self.feature_idx[i] = c_idx
            c_idx += len(self.domains[i])
            if base_instance is not None:
                for c_v, c_c in self.domain_counts[i].items():
                    self.domain_counts[i][c_v] += c_c

        if self.has_missing:
            for c_e in self.examples:
                for i in range(1, self.num_features + 1):
                    if c_e.features[i] == "?":
                        c_e.features[i] = self.domains_max[i]

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
        self.class_distribution[e.cls] += 1
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

    def min_key_random(self, cat_full=False, numeric_full=True):
        supset = []
        features = list(range(1, self.num_features + 1))
        random.shuffle(features)

        # Group examples by classes, if the partitions are non-trivially small, this significantly improves runtime
        classes = defaultdict(list)
        for c_e in self.examples:
            classes[c_e.cls].append(c_e)

        # Check for each pair of examples
        for c_c, c_es in classes.items():
            for c_c2, c_es2 in classes.items():
                if c_c >= c_c2:
                    continue

                for c_e1 in c_es:
                    for c_e2 in c_es2:
                        found = False

                        # Check if there is a disagreement for any feature in the set
                        for c_f, c_v, _ in supset:
                            if c_e1.features[c_f] == "?" or c_e2.features[c_f] == "?":
                                continue

                            if c_f in self.is_categorical:
                                if c_v is None:
                                    if c_e1.features[c_f] != c_e2.features[c_f]:
                                        found = True
                                        break
                                elif (c_e1.features[c_f] == c_v) ^ (c_e2.features[c_f] == c_v):
                                    found = True
                                    break
                            else:
                                if c_v is None:
                                    if c_e1.features[c_f] != c_e2.features[c_f]:
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
                                        if cat_full:
                                            supset.append((c_f, None, None))
                                        else:
                                            supset.append((c_f, c_e1.features[c_f], False))
                                    else:
                                        if numeric_full:
                                            supset.append((c_f, None, None))
                                        else:
                                            supset.append((c_f, min(c_e1.features[c_f], c_e2.features[c_f]), False))
                                    break
                            random.shuffle(features)

        return supset

    def reduce_with_key(self, randomized_runs=3, cat_full=False, numeric_full=False):
        keys = []
        for _ in range(0, randomized_runs):
            keys.append(self.min_key_random(cat_full, numeric_full))

        key = min(keys, key=lambda x: len(x))

        self.reduce(key)

    def test_key(self, key):
        for i in range(0, len(self.examples)):
            for j in range(i + 1, len(self.examples)):
                if self.examples[i].cls != self.examples[j].cls:
                    found = False
                    for c_f, c_v in key:
                        if self.examples[i].features[c_f] == "?" or self.examples[j].features[c_f] == "?":
                            continue
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
        # Trivial decision tree, as a leaf will suffice
        if len(key) == 0:
            return

        if self.reduced_key is not None:
            raise RuntimeError("Instance has already been reduced")

        # This is super slow
        # if not self.test_key(key):
        #     print("Not a key")

        reduce_features = set()
        reduce_thresholds = defaultdict(list)

        for c_f, c_v, c_c in key:
            if len(self.domains[c_f]) > 0:
                if c_v is None:
                    reduce_features.add(c_f)
                    reduce_thresholds[c_f].extend(self.domains[c_f])
                else:
                    try:  # This might happen, if there tree has a threshold that does not occur in the instance
                        if c_c:
                            idx = self.domains[c_f].index(c_v)
                            if idx > 0:
                                reduce_thresholds[c_f].append(self.domains[c_f][idx-1])
                        reduce_features.add(c_f)
                        reduce_thresholds[c_f].append(c_v)
                    except ValueError:
                        pass

        for c_l in reduce_thresholds.values():
            c_l.sort()

        for c_f in reduce_features:
            if c_f not in self.is_categorical:
                if reduce_thresholds[c_f][-1] != self.domains[c_f][-1]:
                    reduce_thresholds[c_f].append(self.domains[c_f][-1])
            else:
                reduce_thresholds[c_f] = list(reduce_thresholds[c_f])
        reduce_features = list(reduce_features)
        self.reduced_map = {f: i+1 for i, f in enumerate(reduce_features)}
        self.reduced_key = key

        known_entries = dict()
        self.reduced_dropped = []
        for c_e in self.examples:
            c_e.original_values = c_e.features
            c_e.features = [None, *(c_e.features[c_f] for c_f in reduce_features)]

            # Map values
            if len(reduce_thresholds) > 0:
                for c_i, c_f in enumerate(reduce_features):
                    if c_e.features[c_i+1] == "?":
                        continue

                    if c_f in self.is_categorical:
                        if c_e.features[c_i+1] not in reduce_thresholds[c_f]:
                            c_e.features[c_i + 1] = "DummyValue"
                    else:
                        for c_v in reduce_thresholds[c_f]:
                            if c_e.features[c_i + 1] <= c_v:
                                c_e.features[c_i + 1] = c_v
                                break

            # Check if can be removed
            values = tuple(c_e.features[1:])

            if values in known_entries:
                if known_entries[values] != c_e.cls:
                    raise RuntimeError("Key is not a real key, duplicate with different classes found.")
                self.reduced_dropped.append(c_e)
                self.class_distribution[c_e.cls] -= 1
            else:
                known_entries[values] = c_e.cls

        # Remove duplicates
        for c_e in reversed(self.reduced_dropped):
            self.examples[c_e.id], self.examples[-1] = self.examples[-1], self.examples[c_e.id]
            self.examples.pop()

        # Update ids
        for i, c_e in enumerate(self.examples):
            c_e.id = i

        self.reduced_original_num_features = self.num_features
        self.reduced_domains = self.domains
        self.reduced_domains_max = self.domains_max
        self.reduced_domain_counts = self.domain_counts
        self.reduced_feature_idx = self.feature_idx
        self.reduced_categorical = self.is_categorical

        self.num_features = len(reduce_features)
        self.domains = [[]]
        self.domains_max = [0]
        self.domain_counts = [None]
        self.feature_idx = {}
        self.is_categorical = set()
        c_idx = 1

        for c_i, c_f in enumerate(reduce_features):
            if len(reduce_thresholds) > 0:
                self.domains.append(reduce_thresholds[c_f])
            else:
                self.domains.append(self.reduced_domains[c_f])
            self.domains_max.append(self.reduced_domains_max[c_f])
            self.domain_counts.append(self.reduced_domain_counts[c_f])
            self.feature_idx[c_i + 1] = c_idx
            c_idx += len(self.domains[c_i+1])
            if c_f in self.reduced_categorical:
                self.is_categorical.add(c_i + 1)

    def unreduce(self, decision_tree=None):
        if self.reduced_map is None:
            return

        self.num_features = self.reduced_original_num_features
        self.domains = self.reduced_domains
        self.domains_max = self.reduced_domains_max
        self.domain_counts = self.reduced_domain_counts
        self.is_categorical = self.reduced_categorical
        self.feature_idx = self.reduced_feature_idx

        for c_e in self.examples:
            c_e.features = c_e.original_values
            c_e.original_values = None
        for c_e in self.reduced_dropped:
            c_e.features = c_e.original_values
            c_e.original_values = None
            c_e.id = len(self.examples)
            self.class_distribution[c_e.cls] += 1
            self.examples.append(c_e)
        self.reduced_dropped = None

        reverse_lookup = {x: y for y, x in self.reduced_map.items()}
        if decision_tree:
            decision_tree.root.remap(reverse_lookup)

        self.reduced_map = None
        self.reduced_key = None
        self.reduced_original_num_features = None
        self.reduced_domains = None
        self.reduced_domains_max = None
        self.reduced_domain_counts = None
        self.reduced_categorical = None
        self.reduced_feature_idx = None

        self.finish()

    def export_c45(self, path, write_names=True, categorical=False):
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
                    if not is_cat and not categorical:
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
    data_file.finish()
    if validation_file is not None:
        validation_file.finish(data_file)
    if test_file:
        test_file.finish(data_file)

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

    return instance

