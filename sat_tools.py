import os
import subprocess
import sys
import resource
from datetime import time
import time


def limit_memory(limit):
    if limit > 0:
        resource.setrlimit(resource.RLIMIT_AS, (limit * 1024 * 1024, (limit + 30) * 1024 * 1024))


class BaseSolver:
    def parse(self, file):
        print("Not implemented")
        raise

    def run(self, input_file, model_file, mem_limit=0):
        print("Not implemented")
        raise

class WrMaxsatSolver(BaseSolver):
    def supports_timeout(self):
        return True

    def run(self, input_file, model_file, timeout=0):
        with open(model_file, "w") as mf:
            if timeout == 0:
                return subprocess.Popen(['bin/uwrmaxsat', input_file, '-m'], stdout=mf)
            else:
                return subprocess.Popen(['bin/uwrmaxsat', input_file, '-m', f'-cpu-lim={timeout}'],
                                        stdout=mf)

    def parse(self, f):
        model = {}
        for _, cl in enumerate(f):
            # Model data
            if cl.startswith("v "):
                values = cl.split(" ")
                for v in values[1:]:
                    converted = int(v)
                    model[abs(converted)] = converted > 0

        if len(model) == 0:
            return None
        return model

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

        p1 = self.solver.run(enc_file, model_file, timeout)

        if timeout == 0 or self.solver.supports_timeout():
            p1.wait()
        else:
            try:
                p1.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if p1.poll() is None:
                    p1.terminate()

        result = None
        with open(model_file, "r") as f:
            model = self.solver.parse(f)
            if model is not None:
                result = inst_encoding.decode(model, instance)

        os.remove(enc_file)
        os.remove(model_file)

        return result
