import bisect
import math
from bisect import bisect_right
from collections import defaultdict

from gmpy2 import popcount

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

    def seed(self, count):
        self.find_next(1 + count)

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
            differences = defaultdict(set)

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
                            for c_f, c_v in differences.keys():
                                if c_f in self.original_instance.is_categorical:
                                    if (e.features[c_f] == c_v) ^ (e2.features[c_f] == c_v):
                                        differences[(c_f, c_v)].add(e2.id)
                                else:
                                    if e.features[c_f] <= c_v < e2.features[c_f] or e.features[c_f] > c_v >= \
                                            e2.features[
                                                c_f]:
                                        differences[(c_f, c_v)].add(e2.id)

                            found = False
                            found_nondiffering = True
                            shuffle(self.features)
                            for c_f in self.features:
                                if e.features[c_f] != e2.features[c_f]:
                                    found = True
                                    if c_f in self.original_instance.is_categorical:
                                        differences[(c_f, e.features[c_f])].add(e2.id)
                                        differences[(c_f, e2.features[c_f])].add(e2.id)
                                    else:
                                        min_v = min(e.features[c_f], e2.features[c_f])
                                        max_v = max(e.features[c_f], e2.features[c_f])

                                        for c_v in self.original_instance.domains[c_f]:
                                            if c_v >= max_v:
                                                break

                                            if min_v <= c_v < max_v:
                                                differences[(c_f, c_v)].add(e2.id)

                            if not found:  # Inconsistent
                                self.possible_examples[-1], self.possible_examples[c_idx] = self.possible_examples[
                                                                                                c_idx], \
                                                                                            self.possible_examples[-1]
                                self.possible_examples.pop()
                                found_nondiffering = False

            if found_nondiffering:
                differing_samples = set(x for t in differences.values() for x in t)

                while len(differing_samples) > 0:
                    k, v = max(differences.items(), key=lambda x: len(x[1] & differing_samples))
                    if k[1] is not None:
                        self.support_set.append((k[0], k[1], False))
                    else:
                        self.support_set.append((k[0], None, None))
                    differing_samples -= v
                    differences.pop(k)

                self.current_instance.add_example(e.copy(self.current_instance))
                self.possible_examples[-1], self.possible_examples[c_idx] = self.possible_examples[c_idx], \
                                                                            self.possible_examples[-1]
                self.possible_examples.pop()
                self.by_class[self.current_instance.examples[-1].cls].append(self.current_instance.examples[-1])
                c_count += 1
            else:
                c_idx += 1

            # None found? Add random sample and start again
            if not found_nondiffering and c_idx >= len(self.possible_examples) and self.possible_examples:
                self.current_instance.add_example(self.possible_examples[-1].copy(self.current_instance))
                self.possible_examples.pop()
                self.by_class[self.current_instance.examples[-1].cls].append(self.current_instance.examples[-1])
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
        return len(self.current_instance.examples) == len(self.original_instance.examples) or len(self.possible_examples) == 0


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

    def seed(self, count):
        self.find_next(1 + count)

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
            differences = defaultdict(set)

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
                            for c_f, c_v in differences.keys():
                                if c_f in self.original_instance.is_categorical:
                                    if (e.features[c_f] == c_v) ^ (e2.features[c_f] == c_v):
                                        differences[(c_f, c_v)].add(e2.id)
                                else:
                                    if e.features[c_f] <= c_v < e2.features[c_f] or e.features[c_f] > c_v >= \
                                            e2.features[
                                                c_f]:
                                        differences[(c_f, c_v)].add(e2.id)

                            found = False
                            found_nondiffering = True
                            shuffle(self.features)
                            for c_f in self.features:
                                if e.features[c_f] != e2.features[c_f]:
                                    found = True
                                    if c_f in self.original_instance.is_categorical:
                                        differences[(c_f, e.features[c_f])].add(e2.id)
                                        differences[(c_f, e2.features[c_f])].add(e2.id)
                                    else:
                                        min_v = min(e.features[c_f], e2.features[c_f])
                                        max_v = max(e.features[c_f], e2.features[c_f])

                                        for c_v in self.original_instance.domains[c_f]:
                                            if c_v >= max_v:
                                                break

                                            if min_v <= c_v < max_v:
                                                differences[(c_f, c_v)].add(e2.id)

                            if not found:  # Inconsistent
                                self.possible_examples[-1], self.possible_examples[c_idx] = self.possible_examples[
                                                                                                c_idx], \
                                                                                            self.possible_examples[-1]
                                self.possible_examples.pop()
                                found_nondiffering = False

            if found_nondiffering:
                differing_samples = set(x for t in differences.values() for x in t)

                while len(differing_samples) > 0:
                    k, v = max(differences.items(), key=lambda x: len(x[1] & differing_samples))
                    if k[1] is not None:
                        self.support_set.append((k[0], k[1], False))
                    else:
                        self.support_set.append((k[0], None, None))
                    differing_samples -= v
                    differences.pop(k)

                self.current_instance.add_example(e.copy(self.current_instance))
                self.possible_examples[-1], self.possible_examples[c_idx] = self.possible_examples[c_idx], \
                                                                            self.possible_examples[-1]
                self.possible_examples.pop()
                self.by_class[self.current_instance.examples[-1].cls].append(self.current_instance.examples[-1])
                c_count += 1
            else:
                c_idx += 1

            # None found? Add random sample and start again
            if not found_nondiffering and c_idx >= len(self.possible_examples) and self.possible_examples:
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

        # Avoid a deadlock if only a few items are left and one class dominates every example
        # i.e. try to have at least one sample for a class
        max_entries = {}
        for c_c in self.original_instance.classes:
            # Is there a sample where this class is dominant anyway
            c_entries = sorted(((distr[c_c], 0 if all(x == c_c for x in distr.keys()) else -1 * max(v for k, v in distr.items() if k != c_c), fi)
                                for fi, (_, distr) in enumerate(count_items) if c_c in distr), reverse=True)
            if len(c_entries) > 0:
                max_entries[c_c] = c_entries

        # Make sure non-dominant entries are represented, if possible
        reserved = []
        non_dominant = set()
        for k in max_entries:
            if all(x[0] <= -x[1] for x in max_entries[k]):
                non_dominant.add(k)
                reserved.append((k, {x[2] for x in max_entries[k]}))

        # Reserve dominant entries, try to avoid entries for non-dominant where possible
        for k in max_entries:
            if k not in non_dominant:
                last_id = None
                for c_cnt, c_oth, c_id in max_entries[k]:
                    c_oth *= -1
                    if c_cnt < c_oth:
                        break

                    if all(len(x) > 1 or c_id not in x for _, x in reserved):
                        last_id = c_id
                        break

                if last_id is None:
                    last_id = max_entries[k][0][2]

                for _, cs in reserved:
                    cs.discard(last_id)

        # Reserve the entries for non-dominant classes
        reserved_entries = {}
        for c_idx in range(0, len(reserved)):
            c_c, entries = reserved[c_idx]

            if len(entries) > 0:
                reserved_id = next(c_id for _, _, c_id in max_entries[c_c] if c_id in entries)
                reserved_entries[reserved_id] = c_c
                for c_idx2 in range(c_idx+1, len(reserved)):
                    reserved[c_idx2][1].discard(reserved_id)

        # Create the actual samples
        for c_idx, (c_features, c_classes) in enumerate(count_items):
            if len(c_classes) > 1:
                self.is_support_set = False

            # Determine the distribution, do not mix different distributions
            if c_idx in reserved_entries:
                cls = reserved_entries[c_idx]
            else:
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
        return self.is_support_set or len(self.original_instance.examples) == len(self.current_instance.examples) \
            or not self.possible_examples


class SupportSetStrategy3:
    def __init__(self, instance):
        self.original_instance = instance
        self.support_set = []
        self.by_class = {x: [] for x in instance.classes}
        self.current_idx = 0
        self.class_idx = 0
        class_examples = defaultdict(list)
        for c_e in instance.examples:
            class_examples[c_e.cls].append(c_e)
        self.class_examples = list(class_examples.values())
        self.features = list((i, i in self.original_instance.is_categorical) for i in range(1, instance.num_features + 1))
        self.changed = True
        self.last_instance = None
        self.feature_thresholds = None
        self.last_cat_defaults = None
        self.feature_map = None
        self.is_support_set = False

    def seed(self, count):
        self.find_next(count)

    def find_next(self, count):
        self.changed = True
        c_count = 0

        while c_count < count and self.class_idx < len(self.class_examples):
            if self.current_idx >= len(self.class_examples[self.class_idx]):
                self.class_idx += 1
                self.current_idx = 0
            if self.class_idx >= len(self.class_examples):
                break

            e = self.class_examples[self.class_idx][self.current_idx]
            found_nondiffering = False
            differences = defaultdict(int)
            comparisons = 0

            for c_class in range(self.class_idx+1, len(self.class_examples)):
                for e2 in self.class_examples[c_class]:
                    if comparisons > 100:
                        break

                    all_nondiffering = True
                    for c_f, c_v, cat in self.support_set:
                        if cat:
                            if (e.features[c_f] == c_v) ^ (e2.features[c_f] == c_v):
                                all_nondiffering = False
                                break
                        else:
                            if e.features[c_f] <= c_v < e2.features[c_f] or e.features[c_f] > c_v >= e2.features[c_f]:
                                all_nondiffering = False
                                break

                    if all_nondiffering:
                        comparisons += 1
                        found = False
                        found_nondiffering = True

                        for c_f, cat in self.features:
                            if e.features[c_f] != e2.features[c_f]:
                                found = True
                                if cat:
                                    differences[(c_f, e.features[c_f])] |= (1 << e2.id)
                                    differences[(c_f, e2.features[c_f])] |= (1 << e2.id)
                                else:
                                    min_v = min(e.features[c_f], e2.features[c_f])
                                    max_v = max(e.features[c_f], e2.features[c_f])
                                    idx = bisect.bisect_left(self.original_instance.domains[c_f], min_v)
                                    idx2 = bisect.bisect_left(self.original_instance.domains[c_f], max_v)
                                    differences[(c_f, self.original_instance.domains[c_f][(idx + idx2) // 2])] |= (1 << e2.id)

                        if not found:  # Inconsistent
                            found_nondiffering = False

            if found_nondiffering:
                differing_samples = 0
                for x in differences.values():
                    differing_samples |= x

                while differing_samples != 0:
                    k, v = max(differences.items(), key=lambda x: popcount(x[1] & differing_samples))
                    self.support_set.append((k[0], k[1], k[0] in self.original_instance.is_categorical))

                    differing_samples &= ~v
                    differences.pop(k)

                c_count += 1
            self.current_idx += 1

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

        # Avoid a deadlock if only a few items are left and one class dominates every example
        # i.e. try to have at least one sample for a class
        max_entries = {}
        for c_c in self.original_instance.classes:
            # Is there a sample where this class is dominant anyway
            c_entries = sorted(((distr[c_c], 0 if all(x == c_c for x in distr.keys()) else -1 * max(v for k, v in distr.items() if k != c_c), fi)
                                for fi, (_, distr) in enumerate(count_items) if c_c in distr), reverse=True)
            if len(c_entries) > 0:
                max_entries[c_c] = c_entries

        # Make sure non-dominant entries are represented, if possible
        reserved = []
        non_dominant = set()
        for k in max_entries:
            if all(x[0] <= -x[1] for x in max_entries[k]):
                non_dominant.add(k)
                reserved.append((k, {x[2] for x in max_entries[k]}))

        # Reserve dominant entries, try to avoid entries for non-dominant where possible
        for k in max_entries:
            if k not in non_dominant:
                last_id = None
                for c_cnt, c_oth, c_id in max_entries[k]:
                    c_oth *= -1
                    if c_cnt < c_oth:
                        break

                    if all(len(x) > 1 or c_id not in x for _, x in reserved):
                        last_id = c_id
                        break

                if last_id is None:
                    last_id = max_entries[k][0][2]

                for _, cs in reserved:
                    cs.discard(last_id)

        # Reserve the entries for non-dominant classes
        reserved_entries = {}
        for c_idx in range(0, len(reserved)):
            c_c, entries = reserved[c_idx]

            if len(entries) > 0:
                reserved_id = next(c_id for _, _, c_id in max_entries[c_c] if c_id in entries)
                reserved_entries[reserved_id] = c_c
                for c_idx2 in range(c_idx+1, len(reserved)):
                    reserved[c_idx2][1].discard(reserved_id)

        # Create the actual samples
        for c_idx, (c_features, c_classes) in enumerate(count_items):
            if len(c_classes) > 1:
                self.is_support_set = False

            # Determine the distribution, do not mix different distributions
            if c_idx in reserved_entries:
                cls = reserved_entries[c_idx]
            else:
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
        return self.is_support_set or self.class_idx >= len(self.class_examples)

