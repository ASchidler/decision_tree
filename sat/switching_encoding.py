import sat.depth_avellaneda as enc1
import sat.depth_partition as enc2

_switch_threshold = 10


def lb():
    return 1


def max_instances(num_features, limit):
    if num_features < 20:
        return 50
    if num_features < 35:
        return 40
    return 25


def new_bound(tree, instance):
    if tree is None:
        return 1

    def dfs_find(node, level):
        if node.is_leaf:
            return level
        else:
            return max(dfs_find(node.left, level + 1), dfs_find(node.right, level + 1))

    return dfs_find(tree.root, 0)


def interrupt(s):
    s.interrupt()


def encode(instance, c_bound, slv, opt_size):
    enc = enc2 if c_bound >= _switch_threshold else enc1
    vs = enc.encode(instance, c_bound, slv, opt_size)
    return vs


def _decode(model, instance, best_depth, vs):
    enc = enc2 if best_depth >= _switch_threshold else enc1
    return enc._decode(model, instance, best_depth, vs)


def extend(slv, instance, vs, c_bound, increment, size_limit):
    enc = enc2 if c_bound >= _switch_threshold else enc1
    return enc.extend(slv, instance, vs, c_bound, increment, size_limit)


def estimate_size(instance, depth):
    """Estimates the required size in the number of literals"""

    return enc2.estimate_size(instance, depth) if depth >= _switch_threshold \
        else enc1.estimate_size(instance, depth)


def encode_size(vs, instance, solver, depth):
    return enc2.encode_size(vs, instance, solver, depth) if depth >= _switch_threshold \
        else enc1.encode_size(vs, instance, solver, depth)


def estimate_size_add(instance, dl):
    return enc2.estimate_size_add(instance, dl) if dl >= _switch_threshold \
        else enc1.estimate_size_add(instance, dl)


def increment():
    return 1

