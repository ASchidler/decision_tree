import random
import feature_encoding
import maxsat.maxsat_feature as maxsat_feature


class BddExamples:
    def __init__(self, features, cls, id):
        self.features = list(features)
        self.cls = cls
        self.id = id

    def copy(self):
        return BddExamples(self.features, self.cls, self.id)

    def dist(self, other_instance, limit):
        cnt = 0
        for i in range(1, limit+1):
            if self.features[i] != other_instance.features[i]:
                cnt += 1
        return cnt


class BddInstance:
    def __init__(self):
        self.num_features = None
        self.examples = []
        self.reduce_map = []

    def add_example(self, example):
        if self.num_features is None:
            self.num_features = len(example.features)

        if example.features[0] is not None:
            # if len(example.features) != self.num_features:
            #     print(f"Example should have {self.num_features} features, but has {len(example.features)}")

            # Make 1 based instead of 0 based
            example.features.insert(0, None)
            self.examples.append(example)
        else:
            # if len(example.features) != self.num_features + 1:
            #     print(f"Example should have {self.num_features} features, but has {len(example.features) - 1}")

            self.examples.append(example)

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

    def unreduce_instance(self, tree):
        self.reduce_map.reverse()
        for v1, v2 in self.reduce_map:
            for n in tree.nodes:
                if n is not None and not n.is_leaf:
                    if n.feature == v1:
                        n.feature = v2
            for e in self.examples:
                e.features[v1], e.features[v2] = e.features[v2], e.features[v1]
        self.num_features = len(self.examples[0].features) - 1

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
        while i < len(key):
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

    def check_consistency(self):
        seen = {}

        for e in self.examples:
            values = []
            for i in range(1, self.num_features + 1):
                values.append(e.features[i])
            c_key = tuple(values)
            if c_key in seen and seen[c_key] != e.cls:
                print("Not consistent")
                exit(1)
            else:
                seen[c_key] = e.cls


def reduce(instance, randomized_runs=5, remove=False, optimal=False):
    print(f"Before: {instance.num_features} Features, {len(instance.examples)} examples")

    if not optimal:
        keys = [instance.min_key(randomize=True) for _ in range(0, randomized_runs)]
        keys.append(instance.min_key(randomize=False))
        keys.sort(key=lambda x: len(x))
        min_key = keys[0]
    else:
        # min_key = feature_encoding.compute_features(instance)
        min_key = maxsat_feature.compute_features(instance)

    removal = set(range(1, instance.num_features + 1))
    for k in min_key:
        removal.remove(k)

    unnecessary = instance.find_same_features()
    removal.update(unnecessary)
    removalL = list(removal)

    # Work from back to front to avoid conflicts
    removalL.sort(reverse=True)

    cIdx = instance.num_features
    instance.reduce_map = []
    for u in removalL:
        while cIdx in removal:
            cIdx -= 1

        if cIdx < 1:
            print("Error, need to remove feature, but non left")

        if cIdx > u:
            instance.reduce_map.append((u, cIdx))
            removal.remove(u)
            removal.add(cIdx)
        for e in instance.examples:
            if cIdx > u:
                e.features[u], e.features[cIdx] = e.features[cIdx], e.features[u]
            if remove:
                e.features.pop(cIdx)

    instance.num_features -= len(removal)

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

    print(f"After: {instance.num_features} Features, {len(instance.examples)} examples")


