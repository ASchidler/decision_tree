import sys
from collections import defaultdict
import class_instance


class MaintainingStrategy:
    def __init__(self, instance):
        self.original_instance = instance
        self.instance = class_instance.ClassificationInstance()
        self.instance.classes = set(instance.classes)

        self.feature_distribution = [0 for _ in range(0, instance.num_features+1)]
        self.class_distribution = defaultdict(int)
        self.classes = defaultdict(list)

        for ce in instance.examples:
            self.class_distribution[ce.cls] += 1
            for i in range(1, instance.num_features+1):
                if ce.features[i]:
                    self.feature_distribution[i] += 1
            self.classes[ce.cls].append(ce)

        for i in range(0, instance.num_features + 1):
            self.feature_distribution[i] /= len(instance.examples)

        for k in self.class_distribution:
            self.class_distribution[k] /= len(instance.examples)

        self.current_class_distribution = {c: 0 for c in self.classes.keys()}
        self.current_feature_distribution = {f: 0 for f in range(0, instance.num_features + 1)}
        self.pool = []

    def pop(self):
        popped = self.instance.examples.pop()
        self.pool.append(popped)

    def add_ex(self, cls):
        if len(self.classes[cls]) == 0:
            raise RuntimeError("Class empty")

        new_ex = self.classes[cls].pop()
        self.instance.add_example(new_ex.copy())
        self.instance.examples[-1].id = len(self.instance.examples)
        self.current_class_distribution[new_ex.cls] += 1

        if len(self.classes[new_ex.cls]) == 0:
            self.classes.pop(new_ex.cls)

        for cf in range(1, self.original_instance.num_features + 1):
            if new_ex.features[cf]:
                self.current_feature_distribution[cf] += 1

    def extend(self, n, tree=None):
        while n > 0 and len(self.classes) > 0:
            n -= 1
            if self.pool:
                self.instance.add_example(self.pool.pop())
                continue
            elif len(self.instance.examples) < len(self.classes):
                nxt_cls = next(x for x in self.classes.keys() if self.current_class_distribution[x] == 0)
                self.add_ex(nxt_cls)
            else:
                best_cls = (2, None)
                for cls in self.classes:
                    p_new = self.current_class_distribution[cls] / len(self.instance.examples)
                    best_cls = min(best_cls, (p_new - self.class_distribution[cls], cls))
                best_cls = best_cls[1]

                best_ex = (sys.maxsize, None)
                changes_t = [0]
                changes_f = [0]
                for f in range(1, self.original_instance.num_features+1):
                    changes_t.append((self.current_feature_distribution[f] + 1) / (len(self.instance.examples)+1))
                    changes_f.append(self.current_feature_distribution[f] / (len(self.instance.examples) + 1))

                for idx, ce in enumerate(self.classes[best_cls]):
                    c_val = 0
                    if tree is not None and tree.decide(ce.features) != ce.cls:
                        c_val = -self.instance.num_features

                    for f in range(1, self.original_instance.num_features+1):
                        c_val += self.feature_distribution[f] - (changes_t[f] if ce.features[f] else changes_f[f])
                        if best_ex[0] > c_val:
                            best_ex = (c_val, idx)
                self.classes[best_cls][-1], self.classes[best_cls][best_ex[1]] = self.classes[best_cls][best_ex[1]], self.classes[best_cls][-1]
                self.add_ex(best_cls)
