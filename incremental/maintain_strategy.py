from collections import defaultdict
import class_instance


class MaintainingStrategy:
    def __init__(self, instance):
        self.original_instance = instance
        self.instance = class_instance.ClassificationInstance()

        self.feature_distribution = [0 for _ in range(0, instance.num_features+1)]
        self.class_distribution = defaultdict(int)
        self.classes = defaultdict(list)

        for ce in instance.examples:
            self.class_distribution[ce.cls] += 1
            for i in range(1, len(ce.features)):
                if ce.features[i]:
                    self.feature_distribution[i] += 1
            self.classes[ce.cls].append(ce)

        for i in range(0, instance.num_features + 1):
            self.feature_distribution[i] /= len(instance.examples)

        for k in self.class_distribution:
            self.class_distribution[k] /= len(instance.examples)

        self.current_class_distribution = {c: 0 for c in self.classes.keys()}
        self.current_feature_distribution = {f: 0 for f in range(0, instance.num_features + 1)}

    def extend(self, n):
        def add_ex(new_ex):
            self.instance.add_example(ex.copy())
            self.instance.examples[-1].id = len(self.instance.examples)
            self.current_class_distribution[ex.cls] += 1

            if len(self.classes[ex.cls]) == 0:
                self.classes.pop(ex.cls)

            for cf in range(1, self.original_instance.num_features + 1):
                if new_ex.features[cf]:
                    self.current_feature_distribution[cf] += 1

        while n > 0 and len(self.classes) > 0:
            n -= 1
            if len(self.instance.examples) == 0:
                ex = next(iter(self.classes.values())).pop()
                add_ex(ex)
            else:
                best_cls = (2, None)
                for cls in self.classes:
                    p_new = self.current_class_distribution[cls] / len(self.instance.examples)
                    best_cls = min(best_cls, (p_new - self.class_distribution[cls], cls))
                best_cls = best_cls[1]

                best_ex = (2, None)
                changes_t = [0]
                changes_f = [0]
                for f in range(1, self.original_instance.num_features+1):
                    changes_t.append((self.current_feature_distribution[f] + 1) / (len(self.instance.examples)+1))
                    changes_f.append(self.current_feature_distribution[f] / (len(self.instance.examples) + 1))

                for idx, ce in enumerate(self.classes[best_cls]):
                    c_val = 0
                    for f in range(1, self.original_instance.num_features+1):
                        c_val += self.feature_distribution[f] - (changes_t[f] if ce.features[f] else changes_f[f])
                    best_ex = min(best_ex, (c_val, idx))
                self.classes[best_cls][-1], self.classes[best_cls][best_ex[1]] = self.classes[best_cls][best_ex[1]], self.classes[best_cls][-1]
                best_ex = self.classes[best_cls].pop()
                add_ex(best_ex)
