import os


def binarize(fn, data, dict):
    lines = set()

    with open(fn, "w") as outp:
        for cd in data:
            n_line = ""
            for i in range(0, len(cd)-1):
                for cc in dict[i][cd[i]]:
                    n_line += cc + ","

            # Avoid duplicates and inconsistent lines
            if n_line not in lines:
                lines.add(n_line)
                outp.write(n_line)
                outp.write(dict[len(cd)-1][cd[-1]])
                outp.write(os.linesep)


def readf(fn):
    data = []
    with open(fn) as inf:
        for _, ln in enumerate(inf):
            fields = ln.split(",")
            c_arr = []
            for fd in fields:
                try:
                    val = int(fd)
                    c_arr.append(val)
                except ValueError:
                    try:
                        val = float(fd)
                        c_arr.append(int(val))
                    except ValueError:
                        c_arr.append(fd)
            data.append(c_arr)

    return data


def map_to_bin(data):
    target = len(data[0])
    for c_arr in data:
        if len(c_arr) != target:
            print(f"Data array mismatch {target}/{len(c_arr)}")

    assignments = []
    # Assign binary values
    for idx in range(0, target):
        cnum = 0
        assignment = {}

        for c_arr in data:
            if c_arr[idx] not in assignment:
                assignment[c_arr[idx]] = bin(cnum)[2:]
                cnum += 1

        assignments.append(assignment)

    # Pad to equal length
    for idx in range(0, target):
        max_len = 0
        for cval in assignments[idx].values():
            max_len = max(max_len, len(cval))
        for ckey, cval in assignments[idx].items():
            assignments[idx][ckey] = "0" * (max_len - len(cval)) + cval

    return assignments


def write_names(fn, assignments):
    with open(fn, "w") as outp:
        outp.write(f"0,1.{os.linesep}")
        catt = 1
        for cassgn in assignments[:-1]:
            cval = next(iter(cassgn.values()))
            for _ in cval:
                outp.write(f"att{catt}:0,1.{os.linesep}")
                catt += 1


# Main
for r, d, f in os.walk("c-format"):
    for fn in f:
        print(f"Processing {fn}")
        if fn.endswith(".data"):
            inp_path = os.path.join(r, fn)
            data = readf(inp_path)
            assignments = map_to_bin(data)
            binarize(os.path.join("binary", fn[:-5] + "_bin.data"), data, assignments)
            write_names(os.path.join("binary", fn[:-5] + "_bin.names"), assignments)

        elif fn.endswith(".train"):
            inp_path = os.path.join(r, fn)
            inp2_path = os.path.join(r, fn[-5] + "test")
            data1 = readf(inp_path)
            data2 = readf(inp_path)

            data = list(data1)
            data.extend(data2)
            assignments = map_to_bin(data)

            binarize(os.path.join("binary", fn[:-6] + "_bin.data"), data1, assignments)
            binarize(os.path.join("binary", fn[:-6] + "_bin.test"), data2, assignments)
            write_names(os.path.join("binary", fn[:-6] + "_bin.names"), assignments)

        print(f"Processed {fn}")
