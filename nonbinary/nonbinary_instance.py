class Example:
    def __init__(self, inst, features, cls):
        self.cls = cls
        self.features = [None, *features]
        self.instance = inst
        self.id = None


class ClassificationInstance:
    def __init__(self):
        self.examples = []
        self.domains = []
        self.num_features = -1
        self.classes = set()

    def add_example(self, e):
        if self.num_features == -1:
            self.num_features = len(e.features) - 1
        elif len(e.features) - 1 != self.num_features:
            raise RuntimeError("Examples have different number of features")

        e.id = len(self.examples)
        self.examples.append(e)
        self.classes.add(e.cls)


def parse(filename):
    instance = ClassificationInstance()

    with open(filename, "r") as f:
        for ln in f:
            fields = ln.split(',')
            example = []

            for i, fd in enumerate(fields[:-1]):
                # Not binary values, skip line
                fd = fd.strip()

                try:
                    fd = int(fd)
                except ValueError:
                    try:
                        fd = float(fd)
                    except ValueError:
                        pass
                example.append(fd)

            cls = fields[-1].strip()
            instance.add_example(Example(instance, example, cls))

    return instance

