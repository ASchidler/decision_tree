import psutil
import limits
from threading import Timer
from sys import maxsize, stdout
from pysat.card import ITotalizer, CardEnc, EncType
import time
from collections import defaultdict
from decision_tree import DecisionTree


def check_memory(s, done):
    used = psutil.Process().memory_info().vms
    free_memory = limits.mem_limit - used

    if free_memory < 3000 * 1024 * 1024:
        print("Caught memout")
        interrupt(s, done)

    elif s.glucose is not None:  # TODO: Not a solver independent way to detect if the solver has been deleted...
        Timer(1, check_memory, [s, done]).start()


def interrupt(s, done, set_done=True):
    print("Interrupted")
    if set_done:
        done.append(True)
    s.interrupt()


def mini_interrupt(s):
    # Increments the current bound, without cancelling
    interrupt(s, None, set_done=False)


def run(enc, instance, solver, start_bound=1, timeout=0, ub=maxsize, opt_size=False, check_mem=True, slim=True):
    clb = enc.lb()
    c_bound = max(clb, start_bound)
    best_model = None
    best_depth = None
    interrupted = []

    # Edge case
    if all(len(x) == 0 for x in instance.domains):
        counts = defaultdict(int)
        for e in instance.examples:
            counts[e.cls] += 1
        _, cls = max((v, k) for k, v in counts.items())
        dt = DecisionTree()
        dt.set_root_leaf(cls)
        return dt

    while clb < ub:
        print(f"Running {c_bound}, " + '{:,}'.format(enc.estimate_size(instance, c_bound)))

        with solver() as slv:
            if check_mem:
                check_memory(slv, interrupted)

            timer = None
            try:
                vs = enc.encode(instance, c_bound, slv, opt_size)
                if timeout > 0:
                    timer = Timer(timeout, interrupt, [slv, interrupted])
                    timer.start()
                solved = slv.solve_limited(expect_interrupt=True)
            except MemoryError:
                return best_model
            finally:
                if timer is not None:
                    timer.cancel()

            if interrupted:
                break
            elif solved:
                model = {abs(x): x > 0 for x in slv.get_model()}
                best_model = enc._decode(model, instance, c_bound, vs)
                best_depth = c_bound
                ub = best_model.get_depth()
                c_bound = ub - enc.increment()
            else:
                c_bound += enc.increment()
                clb = c_bound + enc.increment()

    best_extension = None
    if opt_size and best_model and slim:
        # Try to remove extended leaves
        extension_count = best_model.root.get_extended_leaves()
        if extension_count > 0:
            with solver() as slv:
                c_size_bound = extension_count - 1
                solved = True
                try:
                    vs = enc.encode(instance, best_depth, slv)
                    card = enc.encode_extended_leaf_size(vs, instance, slv, best_depth)
                    tot = ITotalizer(card, c_size_bound+1, top_id=vs["pool"].top + 1)
                    slv.append_formula(tot.cnf)
                    slv.add_clause([-tot.rhs[c_size_bound]])
                except MemoryError:
                    return best_model

                if timeout > 0:
                    timer = Timer(timeout, interrupt, [slv, interrupted])
                    timer.start()

                while solved and c_size_bound >= 0:
                    try:
                        print(f"Running extension {c_size_bound}")
                        stdout.flush()
                        solved = slv.solve_limited(expect_interrupt=True)
                    except MemoryError:
                        break

                    if solved:
                        model = {abs(x): x > 0 for x in slv.get_model()}
                        best_model = enc._decode(model, instance, best_depth, vs)
                        best_extension = best_model.root.get_extended_leaves()
                        c_size_bound = best_extension - 1
                        slv.add_clause([-tot.rhs[c_size_bound]])
                    else:
                        break
                if timer is not None:
                    timer.cancel()

    if opt_size and best_model and enc.estimate_size(instance, best_depth) + enc.estimate_size_add(instance, best_depth) < limits.size_limit:
        with solver() as slv:
            if check_mem:
                check_memory(slv, interrupted)
            print('{:,}'.format(enc.estimate_size(instance, best_depth) + enc.estimate_size_add(instance, best_depth)))
            c_size_bound = best_model.root.get_leaves() - 1
            solved = True

            try:
                vs = enc.encode(instance, best_depth, slv, opt_size)
                if best_extension is not None:
                    card = enc.encode_extended_leaf_size(vs, instance, slv, best_depth)
                    slv.append_formula(
                        CardEnc.atmost(card, bound=best_extension, vpool=vs["pool"], encoding=EncType.totalizer).clauses
                    )
                card = enc.encode_size(vs, instance, slv, best_depth)
                tot = ITotalizer(card, c_size_bound+1, top_id=vs["pool"].top + 1)
                slv.append_formula(tot.cnf)
                slv.add_clause([-tot.rhs[c_size_bound]])
            except MemoryError:
                return best_model

            timer = None

            if timeout > 0:
                timer = Timer(timeout, interrupt, [slv, interrupted])
                timer.start()

            while solved and c_size_bound >= 0:
                try:
                    print(f"Running size {c_size_bound}")
                    solved = slv.solve_limited(expect_interrupt=True)
                except MemoryError:
                    break

                if solved:
                    model = {abs(x): x > 0 for x in slv.get_model()}
                    best_model = enc._decode(model, instance, best_depth, vs)
                    c_size_bound = best_model.root.get_leaves() - 1
                    slv.add_clause([-tot.rhs[c_size_bound]])
                else:
                    break

        if timer is not None:
            timer.cancel()

    return best_model

