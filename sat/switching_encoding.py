import sat.depth_avellaneda as enc1
import sat.depth_partition as enc2
from pysat.card import ITotalizer
from sys import maxsize
from threading import Timer
import time

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


def run(instance, solver, start_bound=1, timeout=0, ub=maxsize, opt_size=False):

    c_bound = start_bound
    clb = 1
    best_model = None
    best_depth = None

    while clb < ub:
        print(f"Running {c_bound}")
        print('{:,}'.format(estimate_size(instance, c_bound)))

        with solver() as slv:
            interrupted = []
            enc = enc2 if c_bound >= _switch_threshold else enc1

            try:
                vs = enc.encode(instance, c_bound, slv)
            except MemoryError:
                return best_model

            if timeout == 0:
                solved = slv.solve()
            else:
                def interrupt(s):
                    s.interrupt()
                    interrupted.append(True)

                timer = Timer(timeout, interrupt, [slv])
                timer.start()
                solved = slv.solve_limited(expect_interrupt=True)
                timer.cancel()

            if interrupted:
                break
            elif solved:
                model = {abs(x): x > 0 for x in slv.get_model()}
                best_model = enc._decode(model, instance, c_bound, vs)
                best_depth = c_bound
                ub = c_bound
                c_bound -= 1
            else:
                c_bound += 1
                clb = c_bound + 1

    if opt_size and best_model:
        with solver() as slv:
            enc = enc2 if best_depth >= _switch_threshold else enc1
            c_size_bound = best_model.root.get_leafs() - 1
            solved = True
            vs = enc.encode(instance, best_depth, slv)
            card = enc.encode_size(vs, instance, slv, best_depth)

            tot = ITotalizer(card, c_size_bound, top_id=vs["pool"].top+1)
            slv.append_formula(tot.cnf)

            timer = None

            if timeout > 0:
                timer = Timer(timeout, interrupt, [slv])
                timer.start()

            while solved:
                print(f"Running {c_size_bound}")
                solved = slv.solve_limited(expect_interrupt=True)

                if solved:
                    model = {abs(x): x > 0 for x in slv.get_model()}
                    best_model = enc._decode(model, instance, best_depth, vs)
                    c_size_bound -= 1
                    slv.add_clause([-tot.rhs[c_size_bound]])
                else:
                    break

        if timer is not None:
            timer.cancel()

    return best_model


def run_incremental(instance, solver, strategy, timeout, size_limit, start_bound=1, increment=5, ubound=maxsize):
    start_time = time.time()
    best_model = enc1.run_incremental(instance, solver, strategy, timeout, size_limit, start_bound, increment, _switch_threshold - 1)
    elapsed = time.time() - start_time

    best_model2 = None
    if elapsed < timeout:
        best_model2 = enc2.run_incremental(instance, solver, strategy, timeout-elapsed, size_limit, _switch_threshold, increment)

    if best_model2 is not None:
        return best_model2

    return best_model


def estimate_size(instance, depth):
    """Estimates the required size in the number of literals"""

    return enc2.estimate_size(instance, depth) if depth >= _switch_threshold \
        else enc1.estimate_size(instance, depth)
