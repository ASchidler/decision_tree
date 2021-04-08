from pysat.formula import CNF, IDPool


class BaseEncoding:
    def __init__(self):
        # This limits avoids the creation of too large instances. This limits the number of literals.
        # Note that each clause in the CNF class is encoded as a list of strings, therefore
        # there is no clear mapping between number of literals as bytes, as the size depends on the sign
        # and how large the variable id is.
        self.size_limit = 1 * 1000 * 1000 * 1000
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

