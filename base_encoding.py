class BaseEncoding:
    def __init__(self, stream):
        self.vars = 0
        self.clauses = 0
        self.stream = stream
        self.increment = 1

    def add_var(self):
        self.vars += 1
        return self.vars

    def add_clause(self, *args):
        if len(args) > 0:
            self.stream.write(' '.join([str(x) for x in args]))
            self.stream.write(" 0\n")
            self.clauses += 1

    '''Encode the conjunction of args as auxiliary variable. 
            The auxiliary variable can then be used instead of the conjunction'''
    def add_auxiliary(self, *args):
        v = self.add_var()
        clause = [v]

        # auxiliary variables
        for a in args:
            self.add_clause(-v, a)
            clause.append(-a)

        self.add_clause(*clause)

        return v

