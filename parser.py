from bdd_instance import BddInstance, BddExamples


def parse(filename):
    instance = BddInstance()
    id = 1
    mappings = {}
    first_line = True

    with open(filename, "r") as f:
        for ln in f:
            # Skip header
            if first_line:
                first_line = False
                continue
            fields = ln.split(',')
            example = []

            for i, fd in enumerate(fields):
                # Not binary values, skip line
                fd = fd.strip()
                estr = fd.lower()

                if i not in mappings:
                    mappings[i] = [estr]

                if mappings[i][0] == estr:
                    example.append(True)
                else:
                    if len(mappings[i]) == 1:
                        mappings[i].append(estr)
                        example.append(False)
                    elif mappings[i][1] == estr:
                        example.append(False)
                    else:
                        print(f"Parsing error for field {i}, found more than 2 values ({mappings[i][0]}, {mappings[i][1]}, {estr})")

            cls = example.pop()
            instance.add_example(BddExamples(example, cls, id))
            id += 1

    return instance