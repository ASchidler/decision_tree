from collections import defaultdict
from bdd_instance import BddInstance
import random
import sys
from decision_tree import DecisionTree

# TODO: Choose k minimal distance sets of l elements, where the distance between the k sets is maximal
# TODO: Award point if better or new is worse
# TODO: Deduct point if worse or new is better
# TODO: Start with smaller sets and build them up.


class InitialStrategy:
    def __init__(self, instance):
        self.instance = instance

    def find_next(self, c_tree, last_tree, last_instance, target, improved, best_instance):
        new_instance = BddInstance()
        new_instance.num_features = self.instance.num_features
        if len(self.instance.examples) <= target:
            for e in self.instance.examples:
                new_instance.add_example(e.copy())
            return new_instance

        p_examples = []
        n_examples = []
        p_rank = []
        n_rank = []

        for e in self.instance.examples:
            if e.cls:
                p_examples.append(e)
            else:
                n_examples.append(e)

        for c_coll, r_coll in [(p_examples, p_rank), (n_examples, n_rank)]:
            for e1 in c_coll:
                dist = 0
                for e2 in c_coll:
                    dist += e1.dist(e2, self.instance.num_features)
                r_coll.append((dist, e1))
            r_coll.sort(key=lambda x: x[0], reverse=True)

        for c_coll in [p_rank, n_rank]:
            for i in range(0, min(target // 2, len(c_coll))):
                new_instance.add_example(c_coll[i][1].copy())

        return new_instance


class InitialStrategy2:
    def __init__(self, instance):
        self.instance = instance

    def find_next(self, c_tree, last_tree, last_instance, target, improved, best_instance):
        new_instance = BddInstance()
        new_instance.num_features = self.instance.num_features

        p_examples = []
        n_examples = []

        if len(self.instance.examples) <= target:
            for e in self.instance.examples:
                new_instance.add_example(e.copy())
            return new_instance

        for e in self.instance.examples:
            if e.cls:
                p_examples.append(e)
            else:
                n_examples.append(e)

        added = set()
        # Initialize with random examples
        new_instance.add_example(p_examples[random.randint(0, len(p_examples) - 1)].copy())
        new_instance.add_example(n_examples[random.randint(0, len(n_examples) - 1)].copy())
        added.add(new_instance.examples[0].id)
        added.add(new_instance.examples[1].id)

        # Add examples with max dist
        while len(new_instance.examples) < target:
            for coll in [p_examples, n_examples]:
                c_max = sys.maxsize
                c_max_examp = None
                for ex in coll:
                    if ex.id not in added:
                        dist = 0
                        for ex2 in new_instance.examples:
                            dist += ex2.dist(ex, new_instance.num_features)
                        if dist < c_max:
                            c_max = dist
                            c_max_examp = ex

                # This may happen if one collection does not have enough entries
                if c_max_examp is not None:
                    added.add(c_max_examp.id)
                    new_instance.add_example(c_max_examp.copy())

        return new_instance


class IncrementalStrategy:
    def __init__(self, instance):
        self.instance = instance
        #self.default_strategy = InitialStrategy(instance)
        self.default_strategy = RandomStrategy(instance)
        #self.default_strategy = NewNewStrategy(instance)
        self.hit_count = {x.id: 0 for x in instance.examples}

    def find_next(self, c_tree, last_tree, last_instance, target, improved, best_instance):
        if c_tree is None:
            return self.default_strategy.find_next(c_tree, last_tree, last_instance, target, improved, best_instance)

        new_instance = BddInstance()
        new_instance.num_features = self.instance.num_features

        if len(self.instance.examples) <= target:
            for e in self.instance.examples:
                new_instance.add_example(e.copy())
            return new_instance

        path_partition_correct = defaultdict(list)
        path_partition_incorrect = defaultdict(list)
        fillers = []

        for e in self.instance.examples:  # last_instance.examples:
            pth = tuple(c_tree.get_path(e.features))
            #result = last_tree.decide(e.features)
            result = c_tree.decide(e.features)

            if result != e.cls:
                path_partition_incorrect[pth].append(e)
            else:
                path_partition_correct[pth].append(e)

        print(f"Found {len(path_partition_correct)} correct paths and {len(path_partition_incorrect)} incorrect paths")
        # Select path representatives
        for k, v in path_partition_correct.items():
            if len(new_instance.examples) > 0:
                v.sort(key=lambda x: (sum(x.dist(ce, self.instance.num_features) for ce in new_instance.examples), -1 * self.hit_count[x.id]))
            else:
                v.sort(key=lambda x: -1 * self.hit_count[x.id])

            c_experiment = v.pop()
            fillers.extend(v)
            v.clear()
            v.append(c_experiment)
            new_instance.add_example(c_experiment.copy())
            self.hit_count[c_experiment.id] += 1

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
            found_any = False
            for k, v in path_partition_incorrect.items():
                if v:
                    _, c_experiment = v.pop()
                    new_instance.add_example(c_experiment.copy())
                    self.hit_count[c_experiment.id] += 1
                    found_any = True

                    if len(new_instance.examples) >= target:
                        break
            if not found_any:
                break

        # Fill up with other examples, if not enough negative examples exist
        if len(new_instance.examples) < target:
            fillers.sort(key=lambda x: -1 * self.hit_count[x.id])
            for i in range(0, min(len(fillers), target - len(new_instance.examples))):
                new_instance.add_example(fillers[i].copy())

        return new_instance


class RandomStrategy:
    def __init__(self, instance):
        self.instance = instance

    def find_next(self, c_tree, last_tree, last_instance, target, improved, best_instance):
        new_instance = BddInstance()
        new_instance.num_features = self.instance.num_features
        if len(self.instance.examples) <= target:
            for e in self.instance.examples:
                new_instance.add_example(e.copy())
        else:
            for _ in range(0, target):
                idx = random.randint(0, len(self.instance.examples) - 1)
                new_instance.add_example(self.instance.examples[idx].copy())

        return new_instance


class RetainingStrategy:
    def __init__(self, instance):
        self.instance = instance
        self.retain = []
        self.default_strategy = InitialStrategy(instance) #RandomStrategy(instance)

    def find_next(self, c_tree, last_tree, last_instance, target, improved, best_instance):
        if c_tree is None:
            return self.default_strategy.find_next(c_tree, last_tree, last_instance, target, improved, best_instance)

        new_instance = BddInstance()
        new_instance.num_features = self.instance.num_features

        if len(self.instance.examples) <= target:
            for e in self.instance.examples:
                new_instance.add_example(e.copy())
            return new_instance

        if best_instance is not None:
            standins = set()
            for r in self.retain:
                standins.add(tuple(last_tree.get_path(r.features)))
                new_instance.add_example(r.copy())

            for e in best_instance.examples:
                pth = tuple(last_tree.get_path(e.features))
                if pth not in standins:
                    standins.add(pth)
                    self.retain.append(e.copy())
                    new_instance.add_example(e.copy())

            print(f"Retained {len(new_instance.examples)} examples")

        # rest = self.default_strategy.find_next(c_tree, last_tree, last_instance, target - len(self.retain))
        # for e in rest.examples:
        #     new_instance.add_example(e)

        neg_examples = defaultdict(list)
        for e in self.instance.examples:
            if c_tree.decide(e.features) != e.cls:
                neg_examples[tuple(last_tree.get_path(e.features))].append(e)

        for _ in range(0, target - len(new_instance.examples)):
            for k, v in neg_examples.items():
                if len(v) > 0:
                    idx = random.randint(0, len(v) - 1)
                    v[idx], v[-1] = v[-1], v[idx]
                    new_instance.add_example(v.pop().copy())
                    if len(new_instance.examples) >= target:
                        break

        return new_instance


class UpdatedRetainingStrategy:
    def __init__(self, instance):
        self.instance = instance
        self.retain = []
        self.default_strategy = NewNewStrategy(instance) # InitialStrategy2(instance) #InitialStrategy(instance) #RandomStrategy(instance)
        self.points = [0 for _ in range(0, max(x.id for x in self.instance.examples) + 1)]

    def find_next(self, c_tree, last_tree, last_instance, target, improved, best_instance):
        if c_tree is None:
            return self.default_strategy.find_next(c_tree, last_tree, last_instance, target, improved, best_instance)

        if last_instance is not None:
            for e in last_instance.examples:
                self.points[e.id] -= 1
        for e in best_instance.examples:
            self.points[e.id] += 1

        new_instance = BddInstance()
        new_instance.num_features = self.instance.num_features

        if len(self.instance.examples) <= target:
            for e in self.instance.examples:
                new_instance.add_example(e.copy())
            return new_instance

        path_partition_correct = defaultdict(list)
        path_partition_incorrect = defaultdict(list)

        added = set()

        def select_instance(input_collection, target_collection, reverse):
            available = [x for x in input_collection if x.id not in added]
            if len(available) == 0:
                return

            max_points = max(self.points[x.id] for x in available)
            top = [x for x in available if self.points[x.id] == max_points]

            if len(top) == 1:
                added.add(top[0].id)
                new_instance.add_example(top[0].copy())
            else:
                top_dist = []
                c_dist = 0
                for c_top in top:
                    for c_other in target_collection:
                        c_dist += c_top.dist(c_other, self.instance.num_features)
                    top_dist.append((c_dist, c_top))
                top_dist.sort(reverse=reverse, key=lambda x: x[0])
                added.add(top_dist[0][1].id)
                new_instance.add_example(top_dist[0][1].copy())

        for e in self.instance.examples:  # last_instance.examples:
            pth = tuple(last_tree.get_path(e.features))
            # result = last_tree.decide(e.features)
            result = c_tree.decide(e.features)

            if result != e.cls:
                path_partition_incorrect[pth].append(e)
            else:
                path_partition_correct[pth].append(e)

        while len(new_instance.examples) < target:
            for pth, examples in path_partition_correct.items():
                if len(new_instance.examples) >= target:
                    break

                select_instance(examples, examples, False)

            for pth, examples in path_partition_incorrect.items():
                if len(new_instance.examples) >= target:
                    break

                target_examples = path_partition_correct[pth] if pth in path_partition_correct else examples
                select_instance(examples, target_examples, False)

        return new_instance


class AAAI:
    def __init__(self, instance):
        self.instance = instance
        self.retain = []

    def find_next(self, c_tree, last_tree, last_instance, target, improved, best_instance):
        new_instance = BddInstance()
        new_instance.num_features = self.instance.num_features
        if c_tree is None:
            for _ in range(0, 5):
                new_instance.add_example(self.instance.examples[random.randint(0, len(self.instance.examples)-1)].copy())

            [self.retain.append(e.copy()) for e in new_instance.examples]

            return new_instance

        new_instance = BddInstance()
        new_instance.num_features = self.instance.num_features

        for e in self.retain:
            new_instance.add_example(e.copy())

        for e in self.instance.examples:
            if c_tree.decide(e.features) != e.cls:
                self.retain.append(e.copy())
                new_instance.add_example(e.copy())
                return new_instance

        return new_instance


class NewNewStrategy:
    def __init__(self, instance):
        self.instance = instance
        # self.default_strategy = InitialStrategy(instance)
        #self.default_strategy = RandomStrategy(instance)
        self.default_strategy = InitialStrategy2(instance)
        self.points = {x.id: 0 for x in instance.examples}

    @staticmethod
    def split(grp, parent, polarity, tree, instance):
        vals = [sys.maxsize]
        c_cls = grp[0].cls
        all_same = True
        for ce in grp:
            if ce.cls != c_cls:
                all_same = False
                break

        if all_same:
            tree.nodes.append(None)
            tree.add_leaf(len(tree.nodes) - 1, parent, polarity, c_cls)
            return

        for f in range(1, instance.num_features + 1):
            c_val = 0
            for ce in grp:
                c_val += 1 if ce.features[f] else -1
            vals.append(abs(c_val))

        new_f, _ = min(enumerate(vals), key=lambda x: x[1])
        if tree.nodes[1] is None:
            tree.set_root(new_f)
        else:
            tree.nodes.append(None)
            tree.add_node(len(tree.nodes) - 1, parent, new_f, polarity)

        # Split groups
        t_grp = []
        f_grp = []
        for ce in grp:
            if ce.features[new_f]:
                t_grp.append(ce)
            else:
                f_grp.append(ce)

        node_id = len(tree.nodes) - 1
        NewNewStrategy.split(t_grp, node_id, True, tree, instance)
        NewNewStrategy.split(f_grp, node_id, False, tree, instance)

    def initialize(self, target):
        new_instance = BddInstance()
        new_instance.num_features = self.instance.num_features

        tree = DecisionTree(self.instance.num_features, 1)

        NewNewStrategy.split(self.instance.examples, None, False, tree, self.instance)

        # Randomize features
        paths = defaultdict(list)
        for e in self.instance.examples:
            paths[tree.get_path(e.features)[-1].id].append(e)

        path_keys = list(paths.keys())
        path_keys.sort(key=lambda x: x % 11)

        while len(new_instance.examples) < target:
            for k in path_keys:
                if paths[k]:
                    new_instance.add_example(paths[k].pop().copy())
                    if len(new_instance.examples) >= target:
                        break

        return new_instance

    def find_next(self, best_tree, worse_tree, worse_instance, target, improved, best_instance):
        if best_tree is None:
            return self.initialize(target)
            #return self.default_strategy.find_next(best_tree, worse_tree, worse_instance, target, improved, best_instance) # self.initialize(target)

        if worse_instance is not None:
            for e in worse_instance.examples:
                self.points[e.id] -= 1

        for e in best_instance.examples:
            self.points[e.id] += 1

        ignore = set(e.id for e in worse_instance.examples) & set(e.id for e in best_instance.examples) if worse_instance is not None else set()

        new_instance = BddInstance()
        new_instance.num_features = self.instance.num_features

        if len(self.instance.examples) <= target:
            for e in self.instance.examples:
                new_instance.add_example(e.copy())
            return new_instance

        path_partition_correct = defaultdict(set)
        path_partition_incorrect = defaultdict(list)
        fillers = []

        for idx, e in enumerate(self.instance.examples):  # last_instance.examples:
            pth = best_tree.get_path(e.features)[-1].id
            result = best_tree.decide(e.features)

            if result != e.cls:
                path_partition_incorrect[pth].append(idx)
            else:
                path_partition_correct[pth].add(idx)

        # Check the features in the tree
        tree_features = set()
        for n in best_tree.nodes:
            if n is not None and not n.is_leaf:
                tree_features.add(n.feature)

        print(f"Found {len(path_partition_correct)} correct paths and {len(path_partition_incorrect)} incorrect paths")

        # Find out which features are non-characteristic
        non_characteristic = {}
        for k, v in path_partition_correct.items():
            c_characteristic = set()
            for f in range(1, self.instance.num_features + 1):
                val = None
                different = False
                for c_idx in v:
                    if val is None:
                        val = self.instance.examples[c_idx].features[f]
                    elif val != self.instance.examples[c_idx].features[f]:
                        different = True
                        break
                if not different:
                    c_characteristic.add(f)
            non_characteristic[k] = c_characteristic

        # Select path representatives
        for k, v in path_partition_correct.items():
            dists = []
            # Calculate distance to previous examples
            for c_idx in v:
                ce = self.instance.examples[c_idx]
                dist = 0
                for te in best_instance.examples:
                    if te.id not in v:
                        #for cf in tree_features:
                        for cf in (x for x in range(1, self.instance.num_features + 1) if x not in tree_features):
                            if ce.features[cf] != te.features[cf]:
                                dist += 1

                # The modifier tries to avoid using the same examples over and over
                modifier = -100 if ce.id in ignore else 0
                # Minimize the distance of non-tree features, i.e. localize the variance on the tree features
                dists.append((self.points[ce.id] + modifier, -1 * dist, c_idx))

            # Put the element with the greatest distance and lowest hit count at the end
            _, _, te = max(dists)
            fillers.extend(dists)
            v.clear()
            v.add(te)
            ce = self.instance.examples[te]

            new_instance.add_example(ce.copy())

        for k, v in path_partition_incorrect.items():
            if k in path_partition_correct:
                representative_id = path_partition_correct[k].pop()
                rep = self.instance.examples[representative_id]

                new_l = []
                for ce in v:
                    dist = 0
                    ce_obj = self.instance.examples[ce]
                    different = set()
                    for f in range(1, self.instance.num_features + 1):
                        if f not in non_characteristic[k]:
                            if rep.features[f] != ce_obj.features[f]:
                                dist += 1
                                different.add(f)
                    #new_l.append((self.points[ce_obj.id], -1 * dist, ce))
                    new_l.append([self.points[ce_obj.id], different, dist, ce])
                path_partition_incorrect[k] = new_l
                #path_partition_incorrect[k].sort()
                path_partition_incorrect[k].sort(key=lambda x: (x[0], x[2]))
            else:
                path_partition_incorrect[k] = [[self.points[self.instance.examples[ce].id], [], 0, ce] for ce in v]
                #path_partition_incorrect[k] = [(self.points[self.instance.examples[ce].id], 0, ce) for ce in v]
                #path_partition_incorrect[k].sort()

        # Select negative representative
        while len(new_instance.examples) < target:
            found_any = False
            for k, v in path_partition_incorrect.items():
                if v:
                    # Find minimum with length over 1
                    c_min = None
                    for itm in v:
                        if len(itm[1]) > 0:
                            # if c_min is None or (len(itm[1]) > 0 and
                            #                          (len(itm[1]) < len(c_min[1]) or
                            #                           (len(itm[1]) == len(c_min[1]) and itm[0] > c_min[0]))):
                            if c_min is None or (len(itm[1]) > 0 and
                                                                           (itm[0] > c_min[0] or
                                                                           (itm[0] == c_min[0] and len(itm[1]) < len(c_min[1])))):
                                c_min = itm
                    if c_min is None:
                        _, _, _, c_idx = v.pop()
                    else:
                        c_idx = c_min[3]
                        v.remove(c_min)
                        for ce in v:
                            ce[1] -= c_min[1]

                    #_, _, c_idx = v.pop()
                    new_instance.add_example(self.instance.examples[c_idx].copy())
                    found_any = True

                    if len(new_instance.examples) >= target:
                        break
            if not found_any:
                break

        # Fill up with other examples
        if len(new_instance.examples) < target:
            fillers.sort()
            while len(new_instance.examples) < target and fillers:
                _, _, c_idx = fillers.pop()
                new_instance.add_example(self.instance.examples[c_idx].copy())

        return new_instance
