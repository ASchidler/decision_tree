import psutil
import limits
from threading import Timer
from sys import maxsize
from pysat.card import ITotalizer


def check_memory(s, done):
    used = psutil.Process().memory_info().vms
    free_memory = limits.mem_limit - used
    print(free_memory)
    if free_memory < 2000 * 1024 * 1024:
        print("Caught")
        done.append(True)
        s.interrupt()
        print("Interrupted")
        print(f"{s.get_status()}")
    elif s.glucose is not None:  # TODO: Not a solver independent way to detect if the solver has been deleted...
        Timer(1, check_memory, [s]).start()


def interrupt(s, done, set_done=True):
    s.interrupt()


def mini_interrupt(s):
    # Increments the current bound
    interrupt(s, None, set_done=False)


def run(enc, instance, solver, start_bound=1, timeout=0, ub=maxsize, opt_size=True, check_mem=True):
    c_bound = start_bound
    clb = 1
    best_model = None
    best_depth = None
    interrupted = []

    while clb < ub:
        print(f"Running {c_bound}")
        print('{:,}'.format(enc.estimate_size(instance, c_bound)))

        with solver() as slv:
            if check_mem:
                check_memory(slv, interrupted)

            try:
                vs = enc.encode(instance, c_bound)
            except MemoryError:
                return best_model
            print("Done")

            timer = None
            if timeout > 0:
                timer = Timer(timeout, interrupt, [slv])
                timer.start()
            solved = slv.solve_limited(expect_interrupt=True)
            if timer is not None:
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
            if check_mem:
                check_memory(slv, interrupted)

            c_size_bound = best_model.root.get_leafs() - 1
            solved = True

            try:
                vs = enc.encode(instance, best_depth, slv)
                card = enc.encode_size(vs, instance, slv, best_depth)
            except MemoryError:
                return best_model

            tot = ITotalizer(card, c_size_bound, top_id=vs["pool"].top+1)
            slv.append_formula(tot.cnf)

            timer = None

            if timeout > 0:
                timer = Timer(timeout, interrupt, [slv])
                timer.start()

            while solved and c_size_bound > 1:
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
