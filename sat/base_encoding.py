from pysat.formula import CNF, IDPool


class BaseEncoding:
    def __init__(self):
        self.size_limit = 15 * 1000 * 1000 * 1000
        self.pool = IDPool()
        self.formula = CNF()
        self.size = 0

    def reset_formula(self):
        self.formula = CNF()
        self.pool = IDPool()
        self.size = 0

    def add_clause(self, args):
        if len(args) > 0:
            self.size += len(args)

            if self.size > self.size_limit:
                raise MemoryError("Encoding size too large")

            self.formula.append(args)

