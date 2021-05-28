import psutil
import limits
from threading import Timer
from sys import maxsize
from pysat.card import ITotalizer
import time


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


def run(enc, instance, solver, start_bound=1, timeout=0, ub=maxsize, opt_size=True, check_mem=True):
    clb = enc.lb()
    c_bound = max(clb, start_bound)
    best_model = None
    best_depth = None
    interrupted = []

    while clb < ub:
        print(f"Running {c_bound}")
        print('{:,}'.format(enc.estimate_size(instance, c_bound)))

        with solver() as slv:
            if check_mem:
                check_memory(slv, interrupted)

            timer = None
            try:
                vs = enc.encode(instance, c_bound, slv, opt_size)
                print("Done Encoding")
                if timeout > 0:
                    timer = Timer(timeout, interrupt, [slv, interrupted])
                    timer.start()
                solved = slv.solve_limited(expect_interrupt=True)
                print("Done solving")
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
                ub = c_bound
                c_bound -= enc.increment()
            else:
                c_bound += enc.increment()
                clb = c_bound + enc.increment()

    if opt_size and best_model and enc.estimate_size(instance, best_depth) + enc.estimate_size_add(instance, best_depth) < limits.size_limit:
        with solver() as slv:
            if check_mem:
                check_memory(slv, interrupted)
            print('{:,}'.format(enc.estimate_size(instance, best_depth) + enc.estimate_size_add(instance, best_depth)))
            c_size_bound = best_model.root.get_leafs() - 1
            solved = True

            try:
                vs = enc.encode(instance, best_depth, slv, opt_size)
                card = enc.encode_size(vs, instance, slv, best_depth)
                print("Done Encoding")
                tot = ITotalizer(card, c_size_bound, top_id=vs["pool"].top + 1)
                slv.append_formula(tot.cnf)
            except MemoryError:
                return best_model

            timer = None

            if timeout > 0:
                timer = Timer(timeout, interrupt, [slv, interrupted])
                timer.start()

            while solved and c_size_bound > 1:
                try:
                    print(f"Running {c_size_bound}")
                    solved = slv.solve_limited(expect_interrupt=True)
                    print("Solved")
                except MemoryError:
                    break

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


def run_incremental(enc, instance, solver, strategy, timeout, size_limit, start_bound=1, increment=5, ubound=maxsize, opt_size=True, check_mem=True):
    start_time = time.time()
    c_bound = start_bound
    best_model = (0.0, None)
    is_done = []
    c_bound = max(c_bound, enc.lb())

    new_best_model = None

    while not is_done and c_bound <= ubound:
        last_runtime = max(5, timeout / 5)
        print(f"Running {c_bound}")
        c_a = 0
        with solver() as slv:
            timer = Timer(timeout-(time.time() - start_time), interrupt, [slv, is_done])
            timer.start()

            if check_mem:
                check_memory(slv, is_done)

            if enc.estimate_size(strategy.instance, c_bound) > size_limit:
                break

            c_guess = enc.estimate_size(strategy.instance, c_bound)

            solved = True
            try:
                vs = enc.encode(strategy.instance, c_bound, slv, opt_size)
            except MemoryError:
                timer.cancel()
                return best_model[1]

            while solved:
                timer2 = Timer(5 * last_runtime, mini_interrupt, [slv])
                timer2.start()
                c_runtime_start = time.time()
                try:
                    solved = slv.solve_limited(expect_interrupt=True)
                except MemoryError:
                    timer.cancel()
                    return best_model[1]

                last_runtime = max(1.0, time.time() - c_runtime_start)

                timer2.cancel()
                if solved:
                    model = {abs(x): x > 0 for x in slv.get_model()}
                    new_best_model = enc._decode(model, strategy.instance, c_bound, vs)
                    c_a = new_best_model.get_accuracy(instance.examples)
                    if best_model[1] is None or best_model[0] < c_a:
                        best_model = (c_a, new_best_model)

                    #print(f"Found: a: {c_a}, d: {new_best_model.get_depth()}, n: {new_best_model.get_nodes()}")
                    if c_a > 0.9999:
                        break

                    strategy.extend(increment, best_model[1])
                    try:
                        result = enc.extend(slv, strategy.instance, vs, c_bound, increment, size_limit - c_guess)
                    except MemoryError:
                        timer.cancel()
                        return best_model[1]

                    if result is None:
                        is_done.append(True)
                        break

                    if result < 0:
                        c_bound -= enc.increment()
                        break

                    c_guess += result

            timer.cancel()
            if c_a > 0.999999:
                is_done.append(True)

            if opt_size and new_best_model is not None and is_done:
                slv.clear_interrupt()
                c_size_bound = new_best_model.root.get_leafs() - 1
                solved = True
                try:
                    card = enc.encode_size(vs, strategy.instance, slv, c_bound)
                    if card is not None:
                        tot = ITotalizer(card, c_size_bound, top_id=vs["pool"].top + 1)
                        slv.append_formula(tot.cnf)
                except MemoryError:
                    return best_model[1]

                timer = Timer(timeout, interrupt, [slv, is_done])
                timer.start()

                while solved:
                    print(f"Running {c_size_bound}")
                    solved = slv.solve_limited(expect_interrupt=True)

                    if solved:
                        model = {abs(x): x > 0 for x in slv.get_model()}
                        new_best_model = enc._decode(model, strategy.instance, c_bound, vs)
                        c_a = new_best_model.get_accuracy(instance.examples)
                        if best_model[1] is None or best_model[0] <= c_a:
                            best_model = (c_a, new_best_model)

                        c_size_bound -= 1
                        slv.add_clause([-tot.rhs[c_size_bound]])
                    else:
                        break
                timer.cancel()
            c_bound += enc.increment()

    return best_model[1]
