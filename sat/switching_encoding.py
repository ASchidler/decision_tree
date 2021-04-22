import sat.depth_avellaneda as enc1
import sat.depth_partition as enc2
from sys import maxsize
from threading import Timer

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


def run(instance, solver, start_bound=1, timeout=0, ub=maxsize):
    c_bound = start_bound
    clb = 0
    best_model = None

    while clb < ub:
        print(f"Running {c_bound}")
        print('{:,}'.format(estimate_size(instance, c_bound)))

        with solver() as slv:
            enc = enc2 if c_bound >= _switch_threshold else enc1

            vs = enc.encode(instance, c_bound, slv)

            if timeout == 0:
                solved = slv.solve()
            else:
                def interrupt(s):
                    s.interrupt()

                timer = Timer(1, interrupt, [slv])
                timer.start()
                solved = slv.solve_limited(expect_interrupt=True)
                timer.cancel()
            if solved:
                model = {abs(x): x > 0 for x in slv.get_model()}
                best_model = enc._decode(model, instance, c_bound, vs)
                ub = c_bound
                c_bound -= 1
            else:
                c_bound += 1
                clb = c_bound + 1

    return best_model


def estimate_size(instance, depth):
    """Estimates the required size in the number of literals"""

    return enc2.estimate_size(instance, depth) if depth >= _switch_threshold \
        else enc1.estimate_size(instance, depth)
