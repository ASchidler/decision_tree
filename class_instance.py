import random
from collections import defaultdict


class ClassificationExample:
    def __init__(self, features, cls, id):
        self.features = list(features)
        self.cls = cls
        self.id = id

    def copy(self):
        return ClassificationExample(self.features, self.cls, self.id)

    def dist(self, other_instance, limit):
        cnt = 0
        for i in range(1, limit+1):
            if self.features[i] != other_instance.features[i]:
                cnt += 1
        return cnt


class ClassificationInstance:
    def __init__(self):
        self.num_features = None
        self.examples = []
        self.reduce_map = []
        self.reduce_features = None
        self.classes = set()

    def add_example(self, example):
        if example.features[0] is not None:
            if self.num_features is None:
                self.num_features = len(example.features)
            # if len(example.features) != self.num_features:
            #     print(f"Example should have {self.num_features} features, but has {len(example.features)}")

            # Make 1 based instead of 0 based
            example.features.insert(0, None)
            self.examples.append(example)
            self.classes.add(example.cls)
        else:
            if self.num_features is None:
                self.num_features = len(example.features) - 1
            # if len(example.features) != self.num_features + 1:
            #     print(f"Example should have {self.num_features} features, but has {len(example.features) - 1}")

            self.examples.append(example)
            self.classes.add(example.cls)

    def find_same_features(self):
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

        return unnecessary

    def unreduce_instance(self, tree, only_tree=False):
        for v1, v2 in self.reduce_map:
            if tree is not None:
                for n in tree.nodes:
                    if n is not None and not n.is_leaf:
                        if n.feature == v1:
                            n.feature = v2
                        elif n.feature == v2:
                            n.feature = v1
            if not only_tree:
                for e in self.examples:
                    e.features[v1], e.features[v2] = e.features[v2], e.features[v1]
        if not only_tree:
            self.num_features = self.reduce_features # len(self.examples[0].features) - 1

    def functional_dependencies(self):
        fd = 0
        unn = 0

        for i in range(1, self.num_features + 1):
            # Find one positively and one negatively classified example
            val = self.examples[0].features[i]
            values1 = {k: v for k, v in enumerate(self.examples[0].features)}
            values2 = None

            for e in self.examples:
                if val != e.features[i]:
                    values2 = {k: v for k, v in enumerate(e.features)}
                    break

            # There is no example that has a different values for feature i
            if values2 is None:
                unn += 1
                continue

            # Build an example set, that requires the full key, or maximizes the number of features in the subset key!
            values1.pop(i)
            values2.pop(i)
            # There is no real feature 0, this in 1-indexed
            values1.pop(0)
            values2.pop(0)

            for e in self.examples:
                if len(values1) == 0:
                    break
                carray = values1 if e.features[i] == val else values2

                for f, fv in list(carray.items()):
                    if fv != e.features[f]:
                        values1.pop(f)
                        values2.pop(f)

            if len(values1) > 0:
                fd += 1

        print(f"Found {fd} FDs and {unn} unnecessary values")

    def min_key(self, randomize=False):
        key = list(range(1, self.num_features + 1))

        if randomize:
            key2 = []
            while key:
                new_idx = random.randint(0, len(key) - 1)
                key2.append(key[new_idx])
                key[-1], key[new_idx] = key[new_idx], key[-1]
                key.pop()
            key = key2

        i = 0
        while i < len(key) and len(key) > 1: # > 1 is necessary, in case the class is the same for all examples
            c_idx = key.pop(i)
            seen_keys = {}
            correct = True

            for e in self.examples:
                values = []
                for j in key:
                    values.append(e.features[j])
                c_key = tuple(values)
                if c_key in seen_keys and seen_keys[c_key] != e.cls:
                    correct = False
                    break
                else:
                    seen_keys[c_key] = e.cls

            if not correct:
                key.insert(i, c_idx)
                i += 1

        # Sanity check
        seen = {}

        for e in self.examples:
            values = []
            for j in key:
                values.append(e.features[j])
            c_key = tuple(values)
            if c_key in seen and seen[c_key] != e.cls:
                print("Not a key")
                exit(1)
            else:
                seen[c_key] = e.cls

        return key

    def min_key2(self):
        """This heuristic takes a different approach and adds the necessary features for distinction one by one"""
        feats = {f: set() for f in range(1, self.num_features+1)}
        key = []

        for e1 in self.examples:
            for e2 in self.examples:
                if e1.cls != e2.cls:
                    for f in range(1, self.num_features+1):
                        if e1.features[f] != e2.features[f]:
                            feats[f].add((e1.id, e2.id))

        cont = True
        while cont:
            cont = False
            max_card = 0
            max_f = None
            for k, v in feats.items():
                if len(v) > max_card:
                    max_card = len(v)
                    max_f = k
            if max_card > 0:
                cont = True
                key.append(max_f)
                vals = feats.pop(max_f)
                for c_v in vals:
                    for v in feats.values():
                        v.discard(c_v)

        seen = {}

        for e in self.examples:
            values = []
            for j in key:
                values.append(e.features[j])
            c_key = tuple(values)
            if c_key in seen and seen[c_key] != e.cls:
                print("Not a key")
                exit(1)
            else:
                seen[c_key] = e.cls

        return key

    def min_key3(self):
        """As min_key2, not greedy but random"""

        key = set()
        for e1 in self.examples:
            for e2 in self.examples:
                if e1.cls != e2.cls:
                    done = False
                    c_diff = []
                    for f in range(1, self.num_features+1):
                        if e1.features[f] != e2.features[f]:
                            if f in key:
                                done = True
                                break
                            c_diff.append(f)
                    # If no differing feature is yet in the key
                    if not done:
                        # Add random feature
                        key.add(c_diff[random.randint(0, len(c_diff) - 1)])

        seen = {}

        for e in self.examples:
            values = []
            for j in key:
                values.append(e.features[j])
            c_key = tuple(values)
            if c_key in seen and seen[c_key] != e.cls:
                print("Not a key")
                exit(1)
            else:
                seen[c_key] = e.cls

        return list(key)

    def check_consistency(self):
        ids = set()
        for i, e1 in enumerate(self.examples):
            ids.add(e1.id)
            for j, e2 in enumerate(self.examples):
                if j > i and e1.cls != e2.cls:
                    found = False
                    for f in range(1, self.num_features+1):
                        if e1.features[f] != e2.features[f]:
                            found = True
                            break
                    if not found:
                        print("Not consistent")
                        exit(1)

    def is_binary(self):
        cls = set()
        for e in self.examples:
            cls.add(e.cls)
            if len(cls) > 2:
                return False

        for f in range(1, self.num_features+1):
            vals = set()
            for e in self.examples:
                vals.add(e.features[f])
                if len(vals) > 2:
                    return False

        return True

    def att_counts(self):
        counts = [0 for _ in range(0, self.num_features + 1)]
        counts.append(0) # class

        for e in self.examples:
            for i in range(0, self.num_features):
                if e.features[i+1]:
                    counts[i+1] += 1
            if e.cls:
                counts[-1] += 1

        return counts


def reduce(instance, randomized_runs=5, remove=False, optimal=False, min_key=None):
    # print(f"Before: {instance.num_features} Features, {len(instance.examples)} examples")

    unnecessary = []
    if min_key is None:
        if not optimal:
            keys = []
            keys.extend([instance.min_key(randomize=True) for _ in range(0, randomized_runs-1)])
            keys.append(instance.min_key(randomize=False))
            keys.extend([instance.min_key3() for _ in range(0, randomized_runs)])
            if len(instance.examples) < 5000 and instance.is_binary():
                keys.append(instance.min_key2())

            keys.sort(key=lambda x: len(x))
            min_key = keys[0]
        else:
            # min_key = feature_encoding.compute_features(instance)
            min_key = maxsat_feature.compute_features(instance)

        unnecessary = instance.find_same_features()

    removal = set(range(1, instance.num_features + 1))
    for k in min_key:
        removal.remove(k)

    removal.update(unnecessary)

    instance.reduce_map = []
    cFront = 1
    cBack = instance.num_features
    instance.reduce_features = instance.num_features

    while cFront < cBack:
        while cFront not in removal and cFront < cBack:
            cFront += 1
        while cBack in removal:
            cBack -= 1
        if cFront < cBack:
            removal.remove(cFront)
            removal.add(cBack)
            instance.reduce_map.append((cFront, cBack))
            for e in instance.examples:
                e.features[cFront], e.features[cBack] = e.features[cBack], e.features[cFront]
                if remove:
                    e.features.pop(cBack)

    instance.num_features -= len(removal)
    instance.num_features = max(instance.num_features, 1)

    # Remove duplicate examples
    seen = set()
    i = 0
    while i < len(instance.examples):
        e = instance.examples[i]
        k = tuple(e.features[f] for f in range(1, instance.num_features + 1))
        if k in seen:
            instance.examples.pop(i)
        else:
            seen.add(k)
            i += 1

    # print(f"After: {instance.num_features} Features, {len(instance.examples)} examples")


def split(instance, ratio_splitoff=0.25):
    cls_ex = defaultdict(list)

    for c_ex in instance.examples:
        cls_ex[c_ex.cls].append(c_ex)

    # Shuffle
    for v in cls_ex.values():
        for idx in range(0, len(v)):
            n_idx = random.randint(idx, len(v) - 1)
            v[idx], v[n_idx] = v[n_idx], v[idx]

    # Establish how many samples to take for each class, to retain the distribution
    overall_target = int(len(instance.examples) * ratio_splitoff)
    cls_target = {k: int(len(v) * ratio_splitoff) for k, v in cls_ex.items()}
    cls_total = sum(v for v in cls_target.values())
    ranking = [(len(v), k) for k, v in cls_ex.items()]
    ranking.sort()

    # Add extra samples to even out rounding errors:
    while cls_total < overall_target:
        for _, k in ranking:
            if cls_total >= overall_target:
                break
            cls_target[k] += 1
            cls_total += 1

    # Create instances
    instance1 = ClassificationInstance()
    instance2 = ClassificationInstance()
    i1_id = 1
    i2_id = 1

    for k, v in cls_target.items():
        for ce in cls_ex[k]:
            nce = ce.copy()
            if v > 0:
                nce.id = i1_id
                instance2.add_example(nce)
                v -= 1
                i1_id += 1
            else:
                nce.id = i2_id
                instance1.add_example(nce)
                i2_id += 1

    return instance1, instance2
