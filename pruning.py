import math
from collections import defaultdict
from decision_tree import DecisionTreeLeaf
from sys import maxsize
from scipy.stats import norm
from sklearn.model_selection import KFold
from class_instance import ClassificationInstance


def prune_reduced_error(tree, pruning_instance, subtree_raise=True):
    # TODO: Add sub-tree raising
    # First remove empty branches
    tree.clean(pruning_instance)
    assigned = tree.assign_samples(pruning_instance)

    def raise_assign(child, samples):
        if child.is_leaf:
            incorrect = 0
            for s in samples:
                if pruning_instance.examples[s].cls != child.cls:
                    incorrect += 1
            return incorrect
        else:
            l_samples = []
            r_samples = []

            for s in samples:
                if pruning_instance.examples[s].features[child.feature]:
                    l_samples.append(s)
                else:
                    r_samples.append(s)

            l_result = raise_assign(child.left, l_samples)
            r_result = raise_assign(child.right, r_samples)
            return l_result + r_result

    def rec_subtree_replace(node):
        if node.is_leaf:
            c_error = 0
            for x in assigned[node.id]:
                x = pruning_instance.examples[x]
                if x.cls != node.cls:
                    c_error += 1
            return c_error, None
        else:
            if len(assigned[node.id]) == 0:
                # Not necessary to classify anything in the pruning set. Simply remove, will propagate a leaf up
                return 0, node.left

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

            # Determine class and error for subtree replacement
            cls = max(clss.items(), key=lambda item: item[1])
            replace_error = sum(y for x, y in clss.items() if x != cls)

            # Calculate errors for subtree raising
            l_raise = maxsize
            r_raise = maxsize
            if subtree_raise:
                l_raise = raise_assign(node.left, assigned[node.id])
                r_raise = raise_assign(node.right, assigned[node.id])

            if old_error <= replace_error and old_error <= l_raise and old_error <= r_raise:
                return old_error, None
            else:
                # Prune
                if replace_error <= l_raise and replace_error <= r_raise:
                    new_leaf = DecisionTreeLeaf(cls, node.id)
                    tree.nodes[node.id] = new_leaf
                    return replace_error, new_leaf
                elif l_raise < r_raise:
                    # TODO: Remove deleted sub-tree from node list
                    return l_raise, node.left
                else:
                    return r_raise, node.right

    rec_subtree_replace(tree.root)


def prune_c45(tree, instance, ratio, m=2, subtree_raise=True):
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
                    new_assigned = {}
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
                    # TODO: Remove implicitly deleted nodes...
                    return c_min[2], c_min[1]
                else:
                    tree.nodes[node.id] = DecisionTreeLeaf(cls, node.id)
                    return [(replace_error, len(assigned[node.id]), replace_e)], tree.nodes[node.id]

    c45_prune_replace_red(tree.root)
    tree.clean(instance, min_samples=m)


def _cost_complexity_alphas(tree, instance):
    errors = {}
    branch_error = {}
    leafs = {}
    p = {}
    classes = {}
    tree = tree.copy()
    assigned = tree.assign_samples(instance)

    def find_errors(node):
        distribution = defaultdict(int)
        for s in assigned[node.id]:
            distribution[instance.examples[s].cls] += 1

        if node.is_leaf:
            incorrect = sum(x[1] for x in distribution.items() if x[0] != node.cls)
            errors[node.id] = incorrect
            branch_error[node.id] = incorrect
            return incorrect, 1

        new_cls, _ = max(distribution.items(), key=lambda k: k[1])
        p[node.left.id] = node
        p[node.right.id] = node
        le, ll = find_errors(node.left)
        re, rl = find_errors(node.right)
        errors[node.id] = sum(x[1] for x in distribution.items() if x[0] != new_cls)
        classes[node.id] = new_cls

        branch_error[node.id] = le + re
        leafs[node.id] = ll + rl

        return le + re, ll + rl

    find_errors(tree.root)

    def do_pass(node):
        if node.is_leaf:
            return maxsize, node.id
        else:
            min_alpha = min(do_pass(node.left), do_pass(node.right))
            rt = errors[node.id] / len(assigned[node.id])
            branch = branch_error[node.id] / len(assigned[node.id])
            c_alpha = (rt - branch) / (leafs[node.id] - 1)

            if c_alpha < min_alpha[0]:
                return c_alpha, node.id
            return min_alpha

    alphas = []
    while True:
        c_min = do_pass(tree.root)
        alphas.append(c_min[0])
        if c_min[1] == tree.root.id:
            break

        # Prune node
        new_node = DecisionTreeLeaf(classes[c_min[1]], c_min[1])
        tree.nodes[c_min[1]] = new_node
        if p[c_min[1]].left.id == c_min[1]:
            p[c_min[1]].left = new_node
        else:
            p[c_min[1]].right = new_node

        diff = errors[c_min[1]] - branch_error[c_min[1]]
        cnode = c_min[1]
        while cnode != tree.root.id:
            cp = p[cnode].id
            leafs[cp] -= leafs[c_min[1]] - 1
            branch_error[cp] += diff
            cnode = cp

    return alphas


def _cost_complexity_prune(tree, instance, test_instance, original_instance, alphas):
    assigned = tree.assign_samples(instance)

    errors = {}
    branch_error = {}
    leafs = {}
    p = {}
    classes = {}
    assigned = tree.assign_samples(instance)

    def find_errors(node):
        distribution = defaultdict(int)
        for s in assigned[node.id]:
            distribution[original_instance.examples[s].cls] += 1

        if node.is_leaf:
            incorrect = sum(x[1] for x in distribution.items() if x[0] != node.cls)
            errors[node.id] = incorrect
            branch_error[node.id] = incorrect
            return incorrect, 1

        new_cls, _ = max(distribution.items(), key=lambda k: k[1])
        p[node.left.id] = node
        p[node.right.id] = node
        le, ll = find_errors(node.left)
        re, rl = find_errors(node.right)
        errors[node.id] = sum(x[1] for x in distribution.items() if x[0] != new_cls)
        classes[node.id] = new_cls

        branch_error[node.id] = le + re
        leafs[node.id] = ll + rl

        return le + re, ll + rl

    find_errors(tree.root)

    def do_pass(node, alpha):
        if node.is_leaf:
            return None, 0, 0
        else:
            l_result = do_pass(node.left, alpha)
            r_result = do_pass(node.right, alpha)

            be_correction = l_result[1] + r_result[1]
            l_correction = l_result[2] + r_result[2]

            branch_error[node.id] += be_correction
            leafs[node.id] -= l_correction

            rt = errors[node.id] / len(assigned[node.id])
            branch = branch_error[node.id] / len(assigned[node.id])
            c_alpha = (rt - branch) / (leafs[node.id] - 1)

            if c_alpha <= alpha:
                new_leaf = DecisionTreeLeaf(classes[node.id], node.id)
                tree.nodes[node.id] = new_leaf
                return new_leaf, errors[node.id] - branch_error[node.id], leafs[node.id] - 1

            if l_result[0] is not None:
                node.left = l_result[0]
            if r_result[0] is not None:
                node.right = r_result[0]
            return None, be_correction, l_correction

    results = []
    for new_alpha in alphas:
        ret = do_pass(tree.root, new_alpha)
        if ret[0] is not None:
            tree.root = ret[0]

        results.append(tree.get_accuracy(test_instance.examples))

    return results


def cost_complexity(tree, instance):
    alphas = _cost_complexity_alphas(tree, instance)

    x = [e.features for e in instance.examples]
    y = [e.cls for e in instance.examples]

    folds = list(KFold().split(x, y))
    accuracies = [0 for _ in alphas]

    for c_fold_training, c_fold_test in folds:
        new_training = ClassificationInstance()
        new_test = ClassificationInstance()

        for c_id in c_fold_training:
            new_training.add_example(instance.examples[c_id])
        for c_id in c_fold_test:
            new_test.add_example(instance.examples[c_id])
        new_tree = tree.copy()
        new_tree.clean(new_training)
        results = _cost_complexity_prune(new_tree, new_training, new_test, instance, alphas)
        for i in range(0, len(accuracies)):
            accuracies[i] += results[i]

    best_alpha, _ = max(list(enumerate(accuracies)), key=lambda k: k[1])
    _cost_complexity_prune(tree, instance, instance, instance, [alphas[best_alpha]])
