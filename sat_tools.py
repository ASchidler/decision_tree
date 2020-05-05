import os
import subprocess
import sys
from enum import Enum


class BaseSolver:
    def parse(self, file):
        print("Not implemented")
        raise

    def run(self, input_file, model_file):
        print("Not implemented")
        raise


class MiniSatSolver(BaseSolver):
    def run(self, input_file, model_file):
        FNULL = open(os.devnull, 'w')
        return subprocess.Popen(['minisat', '-verb=0', input_file, model_file], stdout=FNULL,
                                      stderr=subprocess.STDOUT)

    def parse(self, f):
        first = f.readline()
        if first.startswith("UNSAT"):
            return None

        # TODO: This could be faster using a list...
        model = {}
        vars = f.readline().split()
        for v in vars:
            val = int(v)
            model[abs(val)] = val > 0

        return model


class WrMaxsatSolver(BaseSolver):
    def run(self, input_file, model_file):
        with open(model_file, "w") as mf:
            return subprocess.Popen(['/home/asc/Dev/uwrmaxsat/build/release/bin/uwrmaxsat', input_file, '-m'], stdout=mf)

    def parse(self, f):
        model = {}
        for _, cl in enumerate(f):
            # Model data
            if cl.startswith("v "):
                values = cl.split(" ")
                for v in values[1:]:
                    converted = int(v)
                    model[abs(converted)] = converted > 0

        return model


def add_cardinality_constraint(target_arr, limit, encoder):
    """Limits the cardinality of variables in target_arr <= limit. Expects a BaseEncoder"""

    # Create counter variables
    n = len(target_arr)

    # Add a set of vars for all elements. Set counts how many element have been seen up to this element
    ctr = [[] for _ in range(0, n)]
    for i, c in enumerate(ctr):
        for j in range(0, min(i+1, limit)):
            c.append(encoder.add_var())

    for i in range(1, n-1):
        # Carry over previous element
        for ln in range(0, len(ctr[i-1])):
            encoder.add_clause(-ctr[i-1][ln], ctr[i][ln])

        # Increment counter, if current element is true
        for ln in range(1, len(ctr[i])):
            encoder.add_clause(-ctr[i-1][ln-1], -target_arr[i], ctr[i][ln])

    # Initialize counter on first element
    for i in range(0, n-1):
        encoder.add_clause(-target_arr[i], ctr[i][0])

    # Unsat if counter is exceeded
    for i in range(limit, n):
        encoder.add_clause(-target_arr[i], -ctr[i][limit-1])


class SatRunner:
    def __init__(self, encoder, solver, base_path=".", tmp_file=None):
        self.base_path = base_path
        self.tmp_file = tmp_file if tmp_file is not None else os.getpid()
        self.solver = solver
        self.encoder = encoder

    def run(self, instance, starting_bound, timeout=0):
        l_bound = self.encoder.lb()
        u_bound = sys.maxsize
        c_bound = starting_bound

        enc_file = os.path.join(self.base_path, f"{self.tmp_file}.enc")
        model_file = os.path.join(self.base_path, f"{self.tmp_file}.model")
        out_file = os.path.join(self.base_path, f"{self.tmp_file}.out")

        while l_bound < u_bound:
            print(f"Running with limit {c_bound}")
            with open(enc_file, "w") as f:
                inst_encoding = self.encoder(f)
                inst_encoding.encode(instance, c_bound)

            with open(out_file, "w") as outf:
                p1 = self.solver.run(enc_file, model_file)

            if timeout == 0:
                p1.wait()
            else:
                p1.wait(timeout=timeout)

            with open(model_file, "r") as f:
                model = self.solver.parse(f)
                if model is None:
                    l_bound = c_bound + inst_encoding.increment
                    c_bound = l_bound
                else:
                    tree = inst_encoding.decode(model, instance, c_bound)

                    u_bound = c_bound
                    c_bound -= inst_encoding.increment

            os.remove(enc_file)
            os.remove(model_file)
            os.remove(out_file)

        return tree


class MaxSatRunner:
    def __init__(self, encoder, solver, base_path=".", tmp_file=None):
        self.base_path = base_path
        self.tmp_file = tmp_file if tmp_file is not None else os.getpid()
        self.solver = solver
        self.encoder = encoder

    def run(self, instance, timeout=0):
        enc_file = os.path.join(self.base_path, f"{self.tmp_file}.enc")
        model_file = os.path.join(self.base_path, f"{self.tmp_file}.model")

        with open(enc_file, "w") as f:
            inst_encoding = self.encoder(f)
            inst_encoding.encode(instance)

        p1 = self.solver.run(enc_file, model_file)

        if timeout == 0:
            p1.wait()
        else:
            p1.wait(timeout=timeout)

        result = None
        with open(model_file, "r") as f:
            model = self.solver.parse(f)
            if model is not None:
                result = inst_encoding.decode(model, instance)

        os.remove(enc_file)
        os.remove(model_file)

        return result
