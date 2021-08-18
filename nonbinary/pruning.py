import math
from collections import defaultdict
from collections import deque
from scipy.stats import norm

from nonbinary.decision_tree import DecisionTreeLeaf

c45_default_c = 0.25
c45_default_m = 2


def prune_c45(tree, instance, ratio, m=2, subtree_raise=True):
    tree.clean(instance, min_samples=m)

    z = -1 * norm.ppf(ratio)

    def calc_confidence(n, err):
        f = err / n
        term1 = z * z / (2 * n)
        term2 = z * z / (4 * n * n)
        term_frac = 1 + z * z / n
        return (f + term1 + z * math.sqrt(f / n - f * f / n + term2)) / term_frac

    assigned = tree.assign(instance)

    def c45_reassign(child, samples):
        if child.is_leaf:
            c_classes = defaultdict(int)
            for s in samples:
                c_classes[s.cls] += 1

            if len(samples) > 0:
                best_cls = max(c_classes.items(), key=lambda x: x[1])[0]
                incorrect = sum(x[1] for x in c_classes.items() if x[0] != best_cls)

                e = calc_confidence(len(samples), incorrect)

                return [(incorrect, len(samples), e)]
            else:
                return [(0, 0, 0)]
        else:
            l_samples = []
            r_samples = []

            for s in samples:
                if child._decide(s).id == child.left.id:
                    l_samples.append(s)
                else:
                    r_samples.append(s)

            l_result = c45_reassign(child.left, l_samples)
            r_result = c45_reassign(child.right, r_samples)
            return l_result + r_result

    def c45_prune_replace_red(node):
        if node.is_leaf:
            total = len(assigned[node.id])
            incorrect = 0
            for c_sample in assigned[node.id]:
                if c_sample.cls != node.cls:
                    incorrect += 1
            e = calc_confidence(total, incorrect)

            return [(incorrect, total, e)], None
        else:
            l_result, l_repl = c45_prune_replace_red(node.left)
            r_result, r_repl = c45_prune_replace_red(node.right)
            # Too few samples at leaf
            prune_m = (len(l_result) == 1 and l_result[0][1] < m) or (len(r_result) == 1 and r_result[0][1] < m)
            results = l_result + r_result

            # Commit the pruning reported by children
            if l_repl is not None:
                node.left = l_repl
            if r_repl is not None:
                node.right = r_repl

            # Calculate balanced error of children
            child_e = 0.0
            total = 0
            for i, t, e in results:
                child_e += e * t
                total += t
            child_e /= total

            # Calculate estimated error from subtree replacement
            classes = defaultdict(int)
            for c_sample in assigned[node.id]:
                classes[c_sample.cls] += 1
            cls, _ = max(classes.items(), key=lambda x: x[1])
            replace_error = sum(x[1] for x in classes.items() if x[0] != cls)
            replace_e = calc_confidence(len(assigned[node.id]), replace_error)

            # Compute estimated error of subtree raising
            c_min = None
            if subtree_raise:
                for c_child in [node.left, node.right]:
                    new_incorrect = 0
                    new_total = 0

                    new_results = c45_reassign(c_child, assigned[node.id])

                    for i, t, e in new_results:
                        new_incorrect += i
                        new_total += t
                    new_e = calc_confidence(new_total, new_incorrect)
                    if c_min is None or c_min[0] > new_e:
                        c_min = (new_e, c_child, new_results)

            if replace_e >= child_e and (c_min is None or c_min[0] >= child_e) and not prune_m:
                return results, None
            else:
                # Prune
                if c_min is not None and c_min[0] < replace_e and not prune_m:
                    # TODO: Remove implicitly deleted nodes...
                    # Reclassify should suffice at the end for the whole tree
                    # c_min[1].reclassify([instance.examples[x] for x in assigned[node.id]])
                    return c_min[2], c_min[1]
                else:
                    tree.nodes[node.id] = DecisionTreeLeaf(cls, node.id, tree)
                    return [(replace_error, len(assigned[node.id]), replace_e)], tree.nodes[node.id]

    c45_prune_replace_red(tree.root)
    tree.clean(instance, min_samples=m)


def prune_c45_optimized(tree, instance, validation_tree, validation_training, validation_test, subtree_raise=True, simple=False):
    if simple:
        return prune_c45(tree, instance, c45_default_c, c45_default_m, subtree_raise)

    def get_accuracy(c_val, m_val):
        new_tree = validation_tree.copy()
        new_tree.clean(validation_training, min_samples=m_val)
        prune_c45(new_tree, validation_training, c_val, m_val, subtree_raise)
        acc = new_tree.get_accuracy(validation_test.examples)
        sz = new_tree.get_nodes()
        print(f"f {c_val} {m_val} {sz} {acc}")
        return acc, sz

    # Establish baseline
    best_accuracy, _ = get_accuracy(c45_default_c, c45_default_m)
    best_c = c45_default_c
    best_m = c45_default_m

    max_m = len(validation_training.examples) // 5 * 4
    m_values = [1, 2, 3, 4, *[x for x in range(5, min(50, max_m) + 1, 5)]]

    c_c = 0.01
    while c_c < 0.5:
        last_accuracies = deque(maxlen=5)
        cycle_accuracy = 0
        for c_m in m_values:
            accuracy, new_sz = get_accuracy(c_c, c_m)
            cycle_accuracy = max(cycle_accuracy, accuracy)
            if accuracy < 0.001:
                break

            if new_sz == 1 or (len(last_accuracies) >= 5 and all(x < cycle_accuracy for x in last_accuracies)):
                break

            last_accuracies.append(accuracy)

            if accuracy >= best_accuracy:
                best_c = c_c
                best_m = c_m
                best_accuracy = accuracy

        c_c += 0.01 if c_c < 0.05 else 0.05

    prune_c45(tree, instance, best_c, best_m, subtree_raise)
    tree.root.reclassify(instance.examples)
    return best_accuracy
