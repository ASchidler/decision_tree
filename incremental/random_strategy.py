import math
from collections import defaultdict
from class_instance import ClassificationInstance, ClassificationExample
import random
from sys import maxsize
random.seed(1)


class RandomStrategy:
    def __init__(self, instance, limit=maxsize):
        self.instance = ClassificationInstance()
        self.order = []

        # Find class distribution
        classes = defaultdict(list)

        for c_e in instance.examples:
            classes[c_e.cls].append(c_e)

        for cc in classes.values():
            random.shuffle(cc)

        numbers = {c: 0 for c in classes.keys()}

        while len(self.order) < len(instance.examples) and len(self.order) < limit:
            c_max = max(numbers.items(), key=lambda x: x[1])

            if c_max[0] == 0:
                for k, v in classes:
                    if len(v) == 0:
                        continue
                    c_max = max(c_max, (len(v), k))
                    numbers[k] = len(v) * 100 // len(instance.examples) + 1

            self.order.append(classes[c_max[1]].pop())
            numbers[c_max[1]] -= 1

        self.order.reverse()

    def extend(self, n):
        for _ in range(0, min(n, len(self.order))):
            new_ex = self.order.pop()
            new_ex = new_ex.copy()
            new_ex.id = len(self.instance.examples) + 1
            self.instance.add_example(new_ex)
