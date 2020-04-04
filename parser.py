from BddInstance import BddInstance, BddExamples


def parse(filename):
    instance = BddInstance()
    with open(filename, "r") as f:
        for ln in f:
            fields = ln.split(',')
            example = []
            skip = False

            for fd in fields:
                # Not binary values, skip line
                fd = fd.strip()
                if fd == "1":
                    example.append(True)
                elif fd == "0":
                    example.append(False)
                else:
                    skip = True
                    break

            if not skip:
                cls = example.pop()
                instance.add_example(BddExamples(example, cls))

    return instance