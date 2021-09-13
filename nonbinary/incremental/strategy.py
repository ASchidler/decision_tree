from nonbinary.nonbinary_instance import ClassificationInstance
from random import shuffle


class SupportSetStrategy:
    def __init__(self, instance):
        self.original_instance = instance
        self.support_set = []
        self.current_instance = None
        self.by_class = {x: [] for x in instance.classes}
        self.possible_examples = list(instance.examples[1:])
        self.features = list(range(1, instance.num_features + 1))

    def find_next(self, count):
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
                                if e.features[c_f] == c_v ^ e2.features[c_f] == c_v:
                                    all_nondiffering = False
                                    break
                            else:
                                if e.features[c_f] <= c_v < e2.features[c_f] or e.features[c_f] > c_v >= e2.features[c_f]:
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
                                        self.support_set.append((c_f, min(e.features[c_f], e2.features[c_f]), False))
                                    break

            if found_nondiffering:
                self.current_instance.add_example(e.copy(self.current_instance))
                self.possible_examples[-1], self.possible_examples[c_idx] = self.possible_examples[c_idx], self.possible_examples[-1]
                self.possible_examples.pop()
                self.by_class[self.current_instance.examples[-1].cls].append(self.current_instance.examples[-1])
                c_count += 1
            else:
                c_idx += 1

            # None found? Add random sample and start again
            if not found_nondiffering and c_idx == len(self.possible_examples) - 1:
                self.current_instance.add_example(self.possible_examples[-1].copy(self.current_instance))
                self.possible_examples.pop()
                self.by_class[self.current_instance.examples[-1].cls].append(self.current_instance.examples[-1])
                c_count += 1
                c_idx = 0

        self.current_instance.finish()

