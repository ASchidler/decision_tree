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

    for ci, cl in enumerate(fl):
        if type(cl) is not str:
            cl = cl.decode('ascii')

        # First line
        if ci == 0:
            if cl.startswith("Only "):
                # Error if the index is too high
                return
            data_file = cl.split(",")[0].split(":")[1].strip()
            c_flags = cl[cl.find("(")+1:cl.find(")")].split(",")
            cfs = {x[0]: x[1] for x in (y.strip().split("=") for y in c_flags)}

            if cfs["mode"] == "3":
                flags += "v"
            if cfs["use_dense"] == "True":
                flags += "x"
            if cfs["incremental_strategy"] == "1":
                flags += "a"
            if cfs["encoding"] == "1":
                c_encoding = "a"
            if cfs["categorical"] == "True":
                c_encoding = "c"
            if cfs["encoding"] == "2":
                c_encoding = "y"
            if cfs["encoding"] == "4":
                c_encoding = "p"
            if cfs["encoding"] == "5":
                c_encoding = "z"
            if cfs["size"] == "True":
                flags += "z"
            # if cfs["multiclass"] == "True":
            #     flags += "u"

            c_slice = int(cfs["slice"])

            if cfs["encoding"] == "3":
                c_encoding = "s"
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
            if cfs["use_dt"] == "1":
                flags += "x"
            if cfs["use_dt"] == "2":
                flags += "g"
            if cfs["size_first"] == "True":
                flags += "f"
            if cfs["mode"] == "3":
                flags += "v"
            if cfs["mode"] == "2":
                flags += "j"

            if cfs["mode"] == "0":
                c_algo = "e"
            elif cfs["mode"] == "2":
                c_algo = "j"
            elif cfs["mode"] == "3":
                c_algo = "r"
            else:
                if cfs["heuristic"] == "0":
                    c_algo = "w"
                elif cfs["heuristic"] == "1":
                    c_algo = "c"
                elif cfs["heuristic"] == "2":
                    c_algo = "r"
                # ap.add_argument("-x", dest="use_dense", action="store_true", default=False)
                # ap.add_argument("-a", dest="incremental_strategy", action="store", default=0, type=int, choices=[0, 1])

            if c_encoding == "":
                c_encoding = "0"
            flags = c_val + flags + c_encoding

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

        out_file = f"{data_file}.{c_slice}.{experiment}.{flags}.{c_algo}"
        if len(flags) == 0:
            flags = "0"

        if len(tree_data) > 0:
            out_path = os.path.join("trees", f"{experiment}", out_file+".dt")
            with open(out_path, "w") as out_file:
                out_file.write("".join(tree_data))
        elif depth_lb > 0 or time_taken is not None:
            out_path = os.path.join("trees", f"{experiment}", out_file+".info")
            with open(out_path, "w") as out_file:
                out_file.write(f"{depth_lb}{os.linesep}{size_ub}{os.linesep}{time_taken}{os.linesep}")


with tarfile.open(argv[1]) as tar_file:
    for ctf in tar_file:
        file_parts = ctf.path.split(".")

        if file_parts[-2].startswith("e"):
            # Error file
            continue

        file_parts = file_parts[0].split("-")
        experiment = file_parts[-1]

        cetf = tar_file.extractfile(ctf)
        parse_file(cetf, experiment)
