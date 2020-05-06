from collections import defaultdict
from bdd_instance import BddInstance


class InitialStrategy:
    def __init__(self, instance):
        self.instance = instance

    def find_next(self, c_tree, last_tree, last_instance, target):
        pass


class IncrementalStrategy:
    def __init__(self, instance):
        self.instance = instance
        self.default_strategy = InitialStrategy(instance)
        self.hit_count = {x.id: 0 for x in instance.examples}

    def find_next(self, c_tree, last_tree, last_instance, target):
        if c_tree is None:
            return self.default_strategy.find_next(c_tree, last_tree, last_instance, target)

        new_instance = BddInstance()
        new_instance.num_features = self.instance.num_features

        path_partition_correct = defaultdict(list)
        path_partition_incorrect = defaultdict(list)

        for e in self.instance.examples:  # last_instance.examples:
            pth = tuple(last_tree.get_path(e.features))
            # result = last_tree.decide(e.features)
            result = c_tree.decide(e.features)

            if result != e.cls:
                path_partition_incorrect[pth].append(e)
            else:
                path_partition_correct[pth].append(e)

        print(f"Found {len(path_partition_correct)} correct paths and {len(path_partition_incorrect)} incorrect paths")
        # Select path representatives
        for k, v in path_partition_correct.items():
            v.sort(key=lambda x: self.hit_count[x.id])
            c_experiment = v.pop()
            v.clear()
            v.append(c_experiment)
            new_instance.add_example(c_experiment.copy())

        for k, v in path_partition_incorrect.items():
            if k in path_partition_correct:
                representative = path_partition_correct[k][0]
                path_partition_incorrect[k] = [(representative.dist(ce, self.instance.num_features), ce) for ce in v]
                path_partition_incorrect[k].sort(key=lambda x: (x[0], -1 * self.hit_count[x[1].id]))
            else:
                path_partition_incorrect[k] = [(-1 * self.hit_count[ce.id], ce) for ce in v]
                path_partition_incorrect[k].sort(key=lambda x: x[0])

        # Select negative representative
        while len(new_instance.examples) < target:
            for k, v in path_partition_incorrect.items():
                if v:
                    _, c_experiment = v.pop()
                    new_instance.add_example(c_experiment.copy())
                    self.hit_count[c_experiment.id] += 1

                    if len(new_instance.examples) >= target:
                        break

        return new_instance
