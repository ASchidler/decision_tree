import math
from collections import defaultdict
from class_instance import ClassificationInstance, ClassificationExample
import random
from sys import maxsize
random.seed(1)


class RandomStrategy:
    def __init__(self, instance):
        self.instance = ClassificationInstance()
        self.order = []
        self.instance.classes = set(instance.classes)
        # Find class distribution
        self.classes = defaultdict(list)

        for c_e in instance.examples:
            self.classes[c_e.cls].append(c_e)

        for cc in self.classes.values():
            random.shuffle(cc)

        self.distribution = {k: len(v) / len(instance.examples) for k, v in self.classes.items()}

        self.c_numbers = {c: 0 for c in self.classes.keys()}
        self.c_diffs = {k: v1 for k, v1 in self.distribution.items()}
        self.pool = []

    def extend(self, n, tree=None):
        while n > 0 and len(self.classes) > 0:
            n -= 1
            if self.pool:
                self.instance.add_example(self.pool.pop())
                continue
            elif len(self.instance.examples) < len(self.classes):
                nxt_cls = next(x for x in self.classes.keys() if self.c_numbers[x] == 0)
                new_ex = self.classes[nxt_cls].pop()
                c_max = new_ex.cls
            else:
                c_max, _ = max(self.c_diffs.items(), key=lambda x: x[1])
                new_ex = self.classes[c_max].pop()
            new_ex = new_ex.copy()
            new_ex.id = len(self.instance.examples) + 1
            self.instance.add_example(new_ex)

            if len(self.classes[c_max]) == 0:
                self.classes.pop(c_max)
                self.c_diffs.pop(c_max)
            else:
                self.c_numbers[c_max] += 1
                self.c_diffs[c_max] = self.distribution[c_max] - self.c_numbers[c_max] / len(self.instance.examples)

    def pop(self):
        popped = self.instance.examples.pop()
        self.pool.append(popped)
