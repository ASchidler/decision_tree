from class_instance import ClassificationInstance, ClassificationExample


def parse(filename, has_header=True):
    instance = ClassificationInstance()
    id = 1
    mappings = {}
    first_line = has_header

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
            instance.add_example(ClassificationExample(example, cls, id))

            id += 1

    # Map binary to correct T/F values
    for ce in instance.examples:
        for cf in range(1, len(ce.features)):
            if len(mappings[cf-1]) == 2 and mappings[cf-1][0] == "0" and mappings[cf-1][1] == "1":
                ce.features[cf] = True if not ce.features[cf] else False
        if len(mappings[len(ce.features)-1]) == 2 \
                and mappings[len(ce.features)-1][0] == "0" \
                and mappings[len(ce.features)-1][1] == "1":
            ce.cls = True if not ce.cls else False
    return instance


def parse_nonbinary(filename):
    instance = ClassificationInstance()
    id = 1
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

                example.append(estr)

            cls = example.pop()
            instance.add_example(ClassificationExample(example, cls, id))
            id += 1

    return instance