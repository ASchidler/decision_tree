import base_encoding
import sat_tools
from os import linesep


class MaxsatFeatureEncoding(base_encoding.BaseEncoding):
    def encode(self, instance):
        vars = [self.add_var() if i > 0 else None for i in range(0, instance.num_features + 1)]

        for i in range(0, len(instance.examples)):
            e1 = instance.examples[i]
            for j in range(i+1, len(instance.examples)):
                e2 = instance.examples[j]

                if e1.cls != e2.cls:
                    clause = []
                    for f in range(1, instance.num_features + 1):
                        if e1.features[f] != e2.features[f]:
                            clause.append(vars[f])

                    self.add_clause(instance.num_features, *clause)
        # Soft clauses
        for f in range(1, instance.num_features + 1):
            self.add_clause(1, -vars[f])

        self.write_header(instance)

    def decode(self, model, instance):
        features = []

        for i in range(1, instance.num_features + 1):
            if model[i]:
                features.append(i)

        return features

    @staticmethod
    def lb():
        return 1

    def write_header(self, instance):
        self.stream.seek(0)
        header = f"p wcnf {self.vars} {self.clauses} {instance.num_features + 1}"
        padding = 100 - len(header) - len(linesep)
        self.stream.write(header)
        self.stream.write(" ".join(["" for _ in range(0, padding)]))
        self.stream.write(linesep)


def compute_features(instance):
    runner = sat_tools.MaxSatRunner(MaxsatFeatureEncoding, sat_tools.WrMaxsatSolver())
    result = runner.run(instance, timeout=10)
    return result
