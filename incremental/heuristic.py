import math

import decision_tree as dt
from collections import defaultdict


def compute_tree(instance):
    tree = dt.DecisionTree(instance.num_features, 1)
    c_samples = [(None, None, list(instance.examples))]

    while c_samples:
        idx, pol, es = c_samples.pop()
        first_class = None
        impure = False
        for ce in es:
            if first_class is None:
                first_class = ce.cls
            elif first_class != ce.cls:
                impure = True
                break

        if idx is not None:
            tree.nodes.append(None)

        if not impure:
            if idx is None:
                tree.set_root_leaf(first_class)
            else:
                tree.add_leaf(len(tree.nodes)-1, idx, pol, first_class)
        else:
            # Decide feature
            classes = defaultdict(int)
            features = [[defaultdict(int), defaultdict(int)] for _ in range(0, instance.num_features + 1)]
            for ce in es:
                classes[ce.cls] += 1
                for cf in range(1, instance.num_features + 1):
                    if ce.features[cf]:
                        features[cf][1][ce.cls] += 1
                    else:
                        features[cf][0][ce.cls] += 1

            c_entropy = sum(-(v/len(es)) * math.log2(v/len(es)) for v in classes.values())
            max_id = (-1, None)

            for cf in range(1, instance.num_features + 1):
                ent = 0
                iv = 0
                for cl in features[cf]:
                    tot = sum(v for v in cl.values())
                    if tot == 0:
                        continue
                    ent += tot / len(es) * sum((-v/tot * math.log2(v / tot)) for v in cl.values())
                    iv -= tot / len(es) * math.log2(tot / len(es))

                max_id = max(max_id, ((c_entropy - ent)/iv if iv > 0 else (c_entropy - ent), cf))

            _, best_f = max_id

            # Split
            split_l = []
            split_r = []

            for ce in es:
                if ce.features[best_f]:
                    split_l.append(ce)
                else:
                    split_r.append(ce)

            c_samples.append((len(tree.nodes) - 1, True, split_l))
            c_samples.append((len(tree.nodes) - 1, False, split_r))

            # Set new feature
            if idx is None:
                tree.set_root(best_f)
            else:
                tree.add_node(len(tree.nodes)-1, idx, best_f, pol)

    return tree
