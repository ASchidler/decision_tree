import math
from sys import maxsize
from class_instance import ClassificationInstance
from collections import defaultdict
import random

random.seed(1)


class EntropyStrategy:
    def __init__(self, instance, stratified=False):
        self.original_instance = instance
        self.instance = ClassificationInstance()
        self.instance.classes = set(instance.classes)
        self.stratified = stratified
        self.distribution = [0 for _ in instance.examples[0].features]
        self.distribution_cls = {x: 0 for x in self.original_instance.classes}

        self.feature_entropy = 0

        self.class_remaining = defaultdict(list)
        self.original_distribution_cls = defaultdict(int)
        self.draw_order = []
        for c_e in self.original_instance.examples:
            self.original_distribution_cls[c_e.cls] += 1
            self.class_remaining[c_e.cls].append(c_e)

    def extend(self, n):
        while n > 0 and len(self.class_remaining) > 0:
            n -= 1
            if len(self.instance.examples) < len(self.class_remaining):
                nxt_cls = next(x for x in self.class_remaining.keys() if self.distribution_cls[x] == 0)
                next_example = self.class_remaining[nxt_cls].pop()

                self.distribution_cls[next_example.cls] += 1
                if len(self.class_remaining[next_example.cls]) == 0:
                    self.class_remaining.pop(next_example.cls)
                # Entropy doesnt change since log(1) = 0
            else:
                if not self.stratified:
                    # Decide class by calculating change in entropy
                    max_cls = (-1, None, None)
                    for cls in self.class_remaining.keys():
                        new_p = (self.distribution_cls[cls] + 1) / (len(self.instance.examples) + 1)
                        if new_p > 0:
                            new_p = -1 * new_p * math.log2(new_p)
                        old_p = self.distribution_cls[cls] / (len(self.instance.examples) + 1)
                        if old_p > 0:
                            old_p = -1 * old_p * math.log2(old_p)
                        max_cls = max(max_cls, (new_p - old_p, cls))

                    max_cls = max_cls[1]
                else:
                    # Decide by maintaining original distribution
                    max_cls = (2, None)
                    for cls in self.class_remaining.keys():
                        p = self.original_distribution_cls[cls] / len(self.original_instance.examples)
                        p_new = self.distribution_cls[cls] / len(self.instance.examples)
                        max_cls = min(max_cls, (p_new - p, cls))
                    max_cls = max_cls[1]

                # Calculate example to add
                new_amount = len(self.instance.examples) + 1
                same_entropy_components = []
                added_entropy_components = []
                for i in range(0, len(self.distribution)):
                    # Avoid *= to make the multiplier larger and minimize floating point error
                    p = self.distribution[i] / new_amount
                    p2 = (self.distribution[i] + 1) / new_amount
                    same_entropy_components.append(-1 * p * math.log2(p) if p > 0 else 0)
                    added_entropy_components.append(-1 * p2 * math.log2(p2) if p2 > 0 else 0)

                max_ex = (-1, None)
                c_e_list = self.class_remaining[max_cls]
                for idx, c_e in enumerate(c_e_list):
                    new_entropy = 0
                    for i in range(1, self.original_instance.num_features+1):
                        new_entropy += added_entropy_components[i] if c_e.features[i] else same_entropy_components[i]
                    max_ex = max(max_ex, (new_entropy, idx))

                # Put example at last spot and pop
                c_e_list[max_ex[1]], c_e_list[-1] = c_e_list[-1], c_e_list[max_ex[1]]
                next_example = c_e_list.pop()
                if len(c_e_list) == 0:
                    self.class_remaining.pop(max_cls)

            self.instance.add_example(next_example.copy())
            self.instance.examples[-1].id = len(self.instance.examples)
            self.distribution_cls[next_example.cls] += 1
            for i in range(1, len(next_example.features)):
                if next_example.features[i]:
                    self.distribution[i] += 1


