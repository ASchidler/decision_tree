import base_encoding
import sat_tools


class FeatureEncoding(base_encoding.BaseEncoding):
    def encode(self, instance, limit):
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

                    self.add_clause(*clause)

        sat_tools.add_cardinality_constraint([x for x in vars if x is not None], limit, self)

    def decode(self, model, instance, bound):
        features = []

        for i in range(1, instance.num_features + 1):
            if model[i]:
                features.append(i)

        return features

    @staticmethod
    def lb():
        return 1


def compute_features(instance):
    runner = sat_tools.SatRunner(FeatureEncoding, sat_tools.MiniSatSolver())
    result = runner.run(instance, 1)
    return result
