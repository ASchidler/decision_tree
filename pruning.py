import math
from collections import defaultdict
from decision_tree import DecisionTreeLeaf
from sys import maxsize
from scipy.stats import norm


def prune_reduced_error(tree, pruning_instance):
    # First remove empty branches
    tree.clean(pruning_instance)
    assigned = tree.assign_samples(pruning_instance)

    def rec_subtree_replace(node):
        if node.is_leaf:
            c_error = 0
            for x in assigned[node.id]:
                x = pruning_instance.examples[x]
                if x.cls != node.cls:
                    c_error += 1
            return c_error, None
        else:
            # DFS
            old_error1, repl = rec_subtree_replace(node.left)
            if repl is not None:
                node.left = repl
                tree.nodes[repl.id] = repl
            old_error2, repl = rec_subtree_replace(node.right)
            if repl is not None:
                node.right = repl
                tree.nodes[repl.id] = repl
            old_error = old_error1 + old_error2

            # determine class
            clss = defaultdict(int)
            for s in assigned[node.id]:
                s = pruning_instance.examples[s]
                clss[s.cls] += 1

            if len(clss) == 0:
                return 0, node.left
            else:
                # Determine class and error
                cls = max(clss.items(), key=lambda item: item[1])
                error = sum(y for x, y in clss.items() if x != cls)

            if old_error > error:
                new_leaf = DecisionTreeLeaf(cls, node.id)
                tree.nodes[node.id] = new_leaf
                return error, new_leaf
            else:
                return old_error, None

    rec_subtree_replace(tree.root)

#
# def prune_c45(tree, instance, ratio, m=1):
#     tree.clean(instance, min_samples=m)
#
#     z = -1 * norm.ppf(ratio)
#
#     def calc_confidence(n, err):
#         f = err/n
#         term1 = z * z / (2 * n)
#         term2 = z * z / (4 * n * n)
#         term_frac = 1 + z * z / n
#         return (f + term1 + z * math.sqrt(f/n - f*f/n + term2)) / term_frac
#
#     assigned = tree.assign_samples(instance)
#
#     def c45_prune_replace_red(node):
#         if node.is_leaf:
#             total = len(assigned[node.id])
#             incorrect = 0
#             for c_sample in assigned[node.id]:
#                 if instance.examples[c_sample].cls != node.cls:
#                     incorrect += 1
#             e = calc_confidence(total, incorrect)
#
#             return [(incorrect, total, e)], None
#         else:
#             l_result, l_repl = c45_prune_replace_red(node.left)
#             r_result, r_repl = c45_prune_replace_red(node.right)
#             results = l_result + r_result
#
#             if l_repl is not None:
#                 node.left = l_repl
#             if r_repl is not None:
#                 node.right = r_repl
#
#             classes = defaultdict(int)
#             for c_sample in assigned[node.id]:
#                 classes[instance.examples[c_sample].cls] += 1
#             cls, _ = max(classes.items(), key=lambda x: x[1])
#             new_error = sum(x[1] for x in classes.items() if x[0] != cls)
#
#             child_e = 0.0
#             total = 0
#
#             for i, t, e in results:
#                 child_e += e * t
#                 total += t
#
#             child_e /= total
#             e = calc_confidence(len(assigned[node.id]), new_error)
#
#             if e >= child_e:
#                 return results, None
#             else:
#                 # Prune
#
#
#     c45_prune_replace_red(tree.root)
#     tree.clean(instance, min_samples=m)


def prune_c45(tree, instance, ratio, m=1, subtree_raise=True):
    tree.clean(instance, min_samples=m)

    z = -1 * norm.ppf(ratio)

    def calc_confidence(n, err):
        f = err / n
        term1 = z * z / (2 * n)
        term2 = z * z / (4 * n * n)
        term_frac = 1 + z * z / n
        return (f + term1 + z * math.sqrt(f / n - f * f / n + term2)) / term_frac

    assigned = tree.assign_samples(instance)

    def c45_reassign(child, samples):
        if child.is_leaf:
            incorrect = 0
            for s in samples:
                if instance.examples[s].cls != child.cls:
                    incorrect += 1
            e = calc_confidence(len(samples), incorrect)

            return [(incorrect, len(samples), e)]
        else:
            l_samples = []
            r_samples = []

            for s in samples:
                if instance.examples[s].features[child.feature]:
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
                if instance.examples[c_sample].cls != node.cls:
                    incorrect += 1
            e = calc_confidence(total, incorrect)

            return [(incorrect, total, e)], None
        else:
            l_result, l_repl = c45_prune_replace_red(node.left)
            r_result, r_repl = c45_prune_replace_red(node.right)
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
                classes[instance.examples[c_sample].cls] += 1
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

            if replace_e >= child_e and (c_min is None or c_min[0] >= child_e):
                return results, None
            else:
                # Prune
                if c_min is not None and c_min[0] < replace_e:
                    return c_min[2], c_min[1]
                else:
                    tree.nodes[node.id] = DecisionTreeLeaf(cls, node.id)
                    return [(replace_error, len(assigned[node.id]), replace_e)], tree.nodes[node.id]

    c45_prune_replace_red(tree.root)
    tree.clean(instance, min_samples=m)

def cost_complexity(tree, instance):
    pass
