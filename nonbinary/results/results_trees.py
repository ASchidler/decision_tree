from sys import maxsize, argv
import tarfile
import os


def parse_file(fl, experiment):
    data_file = None
    done = False
    tree_data = []
    flags = ""
    depth_lb = 0
    size_ub = maxsize
    c_slice = None
    c_algo = "c"
    time_taken = None
    c_encoding = ""
    c_val = ""

    start = False
    for ci, cl in enumerate(fl):
        if type(cl) is not str:
            cl = cl.decode('ascii')

        # First line
        if cl.startswith("Only "):
            # Error if the index is too high
            return
        if cl.startswith("Instance"):
            start = True
            data_file = cl.split(",")[0].split(":")[1].strip()
            c_flags = cl[cl.find("(")+1:cl.find(")")].split(",")
            cfs = {x[0]: x[1] for x in (y.strip().split("=") for y in c_flags)}

            if "slim" in cfs:
                if cfs["slim"] == "True":
                    cfs["mode"] = "1"
                else:
                    cfs["mode"] = "0"
                if cfs["weka"] == "False":
                    cfs["heuristic"] = "0"
                else:
                    cfs["heuristic"] = "1"

                if "alt_sat" in cfs and cfs["alt_sat"] == "True":
                    cfs["encoding"] = "1"
                elif "hybrid" in cfs and cfs["hybrid"] == "True":
                    cfs["encoding"] = "2"
                elif "use_smt" in cfs and cfs["use_smt"] == "True":
                    cfs["encoding"] = "3"

            if "encoding" not in cfs:
                cfs["encoding"] = "0"

            flags = cfs["encoding"]
            flags += cfs["mode"]

            if "use_dense" in cfs and cfs["use_dense"] == "True":
                flags += "x"
            if cfs["categorical"] == "True":
                flags += "c"
            if cfs["size"] == "True":
                flags += "z"
            if cfs["mode"] == "2" or cfs["mode"] == "3":
                flags += "a" + cfs["incremental_strategy"]
            # if cfs["multiclass"] == "True":
            #     flags += "u"

            c_slice = int(cfs["slice"])

            if cfs["validation"] == "True":
                c_val = "v"
            if cfs["maintain"] == "True":
                flags += "u"
            if cfs["reduce_numeric"] == "True":
                flags += "n"
            if cfs["reduce_categoric"] == "True":
                flags += "o"
            if cfs["slim_opt"] == "True":
                flags += "e"
            if "use_dt" in cfs and cfs["use_dt"] == "1":
                flags += "x"
            if "use_dt" in cfs and cfs["use_dt"] == "2":
                flags += "g"
            if cfs["size_first"] == "True":
                flags += "f"

            if "mode" in cfs and cfs["mode"] == "0":
                c_algo = "e"
            elif "mode" in cfs and cfs["mode"] == "2":
                c_algo = "j"
            elif "mode" in cfs and cfs["mode"] == "3":
                c_algo = "r"
            else:
                if "heuristic" in cfs and cfs["heuristic"] == "0":
                    c_algo = "w"
                elif "heuristic" in cfs and cfs["heuristic"] == "1":
                    c_algo = "c"
                elif "heuristic" in cfs and cfs["heuristic"] == "2":
                    c_algo = "r"

            flags = c_val + flags + c_encoding
            if len(flags) == 0:
                flags = "0"

            out_file = f"{data_file}.{c_slice}.{experiment}.{flags}.{c_algo}"

            out_path = os.path.join("trees", f"{experiment}", out_file + ".dt")
            # if os.path.exists(out_path):
            #     break

        if done:
            tree_data.append(cl)
        elif cl.startswith("Running depth"):
            depth_lb = int(cl.split(" ")[-1].strip())
        elif cl.startswith("Running size"):
            size_ub = int(cl.split(" ")[-1].strip())
        elif cl.startswith("END Tree"):
            time_taken = cl.split(" ")[-1].strip()
            done = True
        elif cl.startswith("Time: End"):
            done = True

    if len(tree_data) > 0:
        with open(out_path, "w") as of:
            of.write("".join(tree_data))
    if depth_lb > 0 or time_taken is not None:
        out_path = os.path.join("trees", f"{experiment}", out_file+".info")
        with open(out_path, "w") as of:
            of.write(f"{depth_lb}{os.linesep}{size_ub}{os.linesep}{time_taken}{os.linesep}")


with tarfile.open(argv[1]) as tar_file:
    for ctf in tar_file:
        file_parts = ctf.path.split(".")

        if file_parts[-2].startswith("e"):
            # Error file
            continue

        file_parts = file_parts[0].split("-")
        experiment = file_parts[-1][0]

        cetf = tar_file.extractfile(ctf)
        parse_file(cetf, experiment)
