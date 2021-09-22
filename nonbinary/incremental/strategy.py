import math
from bisect import bisect_right
from collections import defaultdict

from nonbinary.nonbinary_instance import ClassificationInstance, Example
from random import shuffle


class SupportSetStrategy:
    def __init__(self, instance):
        self.original_instance = instance
        self.support_set = []
        self.current_instance = None
        self.by_class = {x: [] for x in instance.classes}
        self.possible_examples = list(instance.examples[1:])
        shuffle(self.possible_examples)
        self.features = list(range(1, instance.num_features + 1))
        self.changed = True

    def find_next(self, count):
        self.changed = True
        c_count = 0
        if self.current_instance is None:
            self.current_instance = ClassificationInstance()
            self.current_instance.is_categorical.update(self.original_instance.is_categorical)
            self.current_instance.add_example(self.original_instance.examples[0].copy(self.current_instance))
            c_count += 1
            self.by_class[self.current_instance.examples[0].cls].append(self.current_instance.examples[0])
        else:
            self.current_instance.domains = [set(x) for x in self.current_instance.domains]

        c_idx = 0
        while c_count < count and c_idx < len(self.possible_examples):
            found_nondiffering = False
            e = self.possible_examples[c_idx]
            for c_c, c_elements in self.by_class.items():
                if c_c != e.cls:
                    for e2 in c_elements:
                        all_nondiffering = True
                        for c_f, c_v, _ in self.support_set:
                            if c_f in self.original_instance.is_categorical:
                                if (e.features[c_f] == c_v) ^ (e2.features[c_f] == c_v):
                                    all_nondiffering = False
                                    break
                            else:
                                if e.features[c_f] <= c_v < e2.features[c_f] or e.features[c_f] > c_v >= e2.features[c_f]:
                                    all_nondiffering = False
                                    break

                        if all_nondiffering:
                            found_nondiffering = True
                            found = False
                            shuffle(self.features)
                            for c_f in self.features:
                                if e.features[c_f] != e2.features[c_f]:
                                    found = True
                                    if c_f in self.original_instance.is_categorical:
                                        self.support_set.append((c_f, e.features[c_f], False))
                                    else:
                                        self.support_set.append((c_f, min(e.features[c_f], e2.features[c_f]), False))
                                    break
                            if not found:  # Inconsistent
                                found_nondiffering = False

            if found_nondiffering:
                self.current_instance.add_example(e.copy(self.current_instance))
                self.possible_examples[-1], self.possible_examples[c_idx] = self.possible_examples[c_idx], self.possible_examples[-1]
                self.possible_examples.pop()
                self.by_class[self.current_instance.examples[-1].cls].append(self.current_instance.examples[-1])
                c_count += 1
            else:
                c_idx += 1

            # None found? Add random sample and start again
            if not found_nondiffering and c_idx >= len(self.possible_examples):
                feature_thresholds = defaultdict(list)

                # First ensure consistency
                for f, t, _ in self.support_set:
                    feature_thresholds[f].append(t)
                for f, v in feature_thresholds.items():
                    if f in self.original_instance.is_categorical:
                        feature_thresholds[f] = set(v)
                    else:
                        v.sort()
                        if v[-1] != self.original_instance.domains[f][-1]:
                            v.append(self.original_instance.domains[f][-1])

                features = list(feature_thresholds.keys())
                cls_map = {}
                for c_e in self.current_instance.examples:
                    vals = []
                    for c_f in features:
                        for c_idx, c_t in enumerate(feature_thresholds[c_f]):
                            if c_e.features[c_f] <= c_t:
                                vals.append(c_idx)
                                break
                        cls_map[tuple(vals)] = c_e.cls

                while self.possible_examples:
                    vals = []
                    for c_f in features:
                        for c_idx, c_t in enumerate(feature_thresholds[c_f]):
                            if self.possible_examples[-1].features[c_f] <= c_t:
                                vals.append(c_idx)
                                break
                    if tuple(vals) in cls_map and cls_map[tuple(vals)] != self.possible_examples[-1].cls:
                        self.possible_examples.pop()
                    else:
                        break

                # Add random sample
                if self.possible_examples:
                    self.current_instance.add_example(self.possible_examples[-1].copy(self.current_instance))
                    self.possible_examples.pop()
                    self.by_class[self.current_instance.examples[-1].cls].append(self.current_instance.examples[-1])
                    c_count += 1
                    c_idx = 0

        self.current_instance.finish()

    def get_instance(self):
        if self.changed:
            self.changed = False
            self.current_instance.unreduce()
            self.current_instance.reduce(self.support_set)

        return self.current_instance

    def unreduce(self, tree):
        self.current_instance.unreduce(tree)

    def done(self):
        return len(self.current_instance.examples) == len(self.original_instance.examples)


class SupportSetStrategy2:
    def __init__(self, instance):
        self.original_instance = instance
        self.support_set = []
        self.current_instance = None
        self.by_class = {x: [] for x in instance.classes}
        self.possible_examples = list(instance.examples[1:])
        shuffle(self.possible_examples)
        self.features = list(range(1, instance.num_features + 1))
        self.changed = True
        self.last_instance = None
        self.feature_thresholds = None
        self.last_cat_defaults = None
        self.feature_map = None
        self.is_support_set = False

    def find_next(self, count):
        self.changed = True
        c_count = 0
        if self.current_instance is None:
            self.current_instance = ClassificationInstance()
            self.current_instance.is_categorical.update(self.original_instance.is_categorical)
            self.current_instance.add_example(self.original_instance.examples[0].copy(self.current_instance))
            c_count += 1
            self.by_class[self.current_instance.examples[0].cls].append(self.current_instance.examples[0])
        else:
            self.current_instance.domains = [set(x) for x in self.current_instance.domains]

        c_idx = 0
        while c_count < count and c_idx < len(self.possible_examples):
            found_nondiffering = False
            e = self.possible_examples[c_idx]
            for c_c, c_elements in self.by_class.items():
                if c_c != e.cls:
                    for e2 in c_elements:
                        all_nondiffering = True
                        for c_f, c_v, _ in self.support_set:
                            if c_f in self.original_instance.is_categorical:
                                if (e.features[c_f] == c_v) ^ (e2.features[c_f] == c_v):
                                    all_nondiffering = False
                                    break
                            else:
                                if e.features[c_f] <= c_v < e2.features[c_f] or e.features[c_f] > c_v >= e2.features[
                                    c_f]:
                                    all_nondiffering = False
                                    break

                        if all_nondiffering:
                            found_nondiffering = True
                            shuffle(self.features)
                            for c_f in self.features:
                                if e.features[c_f] != e2.features[c_f]:
                                    if c_f in self.original_instance.is_categorical:
                                        self.support_set.append((c_f, e.features[c_f], False))
                                    else:
                                        target_v = min(e.features[c_f], e2.features[c_f])
                                        right_idx = bisect_right(self.original_instance.domains[c_f], target_v)
                                        if target_v != self.original_instance.domains[c_f][right_idx]:
                                            if right_idx > 0 and target_v == self.original_instance.domains[c_f][right_idx - 1]:
                                                target_v = self.original_instance.domains[c_f][right_idx - 1]
                                            else:
                                                rv = self.original_instance.domains[c_f][right_idx]
                                                lv = self.original_instance.domains[c_f][right_idx - 1]
                                                if e.features[c_f] <= rv < e2.features[c_f] or e.features[c_f] > rv >= e2.features[c_f]:
                                                    target_v = rv
                                                else:
                                                    target_v = lv

                                        self.support_set.append((c_f, target_v, False))
                                    break

            if found_nondiffering:
                self.current_instance.add_example(e.copy(self.current_instance))
                self.possible_examples[-1], self.possible_examples[c_idx] = self.possible_examples[c_idx], \
                                                                            self.possible_examples[-1]
                self.possible_examples.pop()
                self.by_class[self.current_instance.examples[-1].cls].append(self.current_instance.examples[-1])
                c_count += 1
            else:
                c_idx += 1

            # None found? Add random sample and start again
            if not found_nondiffering and c_idx >= len(self.possible_examples):
                self.current_instance.add_example(self.possible_examples[-1].copy(self.current_instance))
                self.possible_examples.pop()
                self.by_class[self.current_instance.examples[-1].cls].append(self.current_instance.examples[-1])
                c_idx = 0

    def get_instance(self):
        if not self.changed:
            return self.last_instance
        self.changed = False

        self.feature_thresholds = defaultdict(list)

        for f, t, _ in self.support_set:
            self.feature_thresholds[f].append(t)
        for f, v in self.feature_thresholds.items():
            if f in self.original_instance.is_categorical:
                self.feature_thresholds[f] = set(v)
            else:
                v.sort()
                if v[-1] != self.original_instance.domains[f][-1]:
                    v.append(self.original_instance.domains[f][-1])

        counts = defaultdict(lambda: defaultdict(int))
        dummy_counts = defaultdict(lambda: defaultdict(int))
        self.last_instance = ClassificationInstance()
        self.feature_map = list(self.feature_thresholds.keys())

        for c_fi, c_f in enumerate(self.feature_map):
            if c_f in self.original_instance.is_categorical:
                self.last_instance.is_categorical.add(c_fi + 1)

        for c_e in self.original_instance.examples:
            representative = []
            for c_f in self.feature_map:
                c_t = self.feature_thresholds[c_f]
                if c_f in self.original_instance.is_categorical:
                    if c_e.features[c_f] in c_t:
                        representative.append(c_e.features[c_f])
                    else:
                        representative.append("DummyValue")
                        dummy_counts[c_f][c_e.features[c_f]] += 1
                else:
                    for i in range(0, len(c_t)):
                        if c_e.features[c_f] <= c_t[i]:
                            representative.append(i)
                            break
            counts[tuple(representative)][c_e.cls] += 1

        self.is_support_set = True

        count_items = list(counts.items())
        for c_features, c_classes in count_items:
            if len(c_classes) > 1:
                self.is_support_set = False

            # Determine the distribution, do not mix different distributions
            c_max = None
            for k, v in c_classes.items():
                if c_max is None or v > c_max[1] or (v == c_max[1] and k not in self.last_instance.classes):
                    c_max = (k, v)
            cls = c_max[0]
            # if len(c_classes) == 1 and False:  # Pure
            #     cls = "p_"+ next(iter(c_classes.keys()))
            # else:
            #     cls = max((v, k) for k, v in c_classes.items())[1]
                # sum_cls = sum(v for k, v in c_classes.items() if k != cls)
                # if sum_cls > 0.3 * c_classes[cls]:  # Balanced
                #     cls = "b_" + cls
                # else:  # low density of different classes
                #     cls = "l_"+ cls

            self.last_instance.add_example(Example(self.last_instance, list(c_features), cls))
            self.last_instance.examples[-1].impurities = c_classes

        # Avoid a deadlock if only a few items are left and one class dominates every example
        for c_c in self.original_instance.classes:
            if len(self.last_instance.classes) >= len(self.last_instance.examples):
                break

            if c_c not in self.last_instance.classes:
                entries = [(v, fi) for fi, (f, cls) in enumerate(count_items) for k, v in cls.items() if k == c_c]
                if len(entries) > 0:
                    _, max_cc = max(entries)
                    self.last_instance.examples[max_cc].cls = c_c
                    self.last_instance.classes.add(c_c)
        self.last_instance.finish()

        self.last_cat_defaults = {}
        for c_f, c_counts in dummy_counts.items():
            self.last_cat_defaults[c_f] = max((v, k) for k, v in c_counts.items())[1]

        return self.last_instance

    def unreduce(self, tree):
        reverse_map = {i+1: f for i, f in enumerate(self.feature_map)}

        for c_n in tree.nodes:
            if c_n and not c_n.is_leaf:
                n_f = reverse_map[c_n.feature]
                if c_n.feature in self.last_instance.is_categorical:
                    if c_n.threshold == "DummyValue":
                        c_n.threshold = self.last_cat_defaults[n_f]
                else:
                    c_n.threshold = self.feature_thresholds[n_f][int(math.ceil(c_n.threshold))]
                c_n.feature = n_f

    def done(self):
        return self.is_support_set or len(self.original_instance.examples) == len(self.current_instance.examples)
