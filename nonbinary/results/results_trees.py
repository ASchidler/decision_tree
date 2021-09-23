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
            for cf in c_flags:
                cfs = cf.strip().split("=")

                if cfs[0] == "mode" and cfs[1] == "3":
                    flags += "v"
                elif cfs[0] == "use_dense" and cfs[1] == "True":
                    flags += "x"
                elif cfs[0] == "incremental_strategy" and cfs[1] == "1":
                    flags += "a"
                if cfs[0] == "encoding" and cfs[1] == "1":
                    c_encoding = "a"
                elif cfs[0] == "categorical" and cfs[1] == "True":
                    c_encoding = "c"
                if cfs[0] == "encoding" and cfs[1] == "2":
                    c_encoding = "y"
                if cfs[0] == "encoding" and cfs[1] == "4":
                    c_encoding = "p"
                if cfs[0] == "encoding" and cfs[1] == "5":
                    c_encoding = "z"
                elif cfs[0] == "size" and cfs[1] == "True":
                    flags += "z"
                elif cfs[0] == "multiclass" and cfs[1] == "True":
                    flags += "u"
                elif cfs[0] == "slice":
                    c_slice = int(cfs[1])
                if cfs[0] == "encoding" and cfs[1] == "3":
                    c_encoding = "s"
                elif cfs[0] == "validation" and cfs[1] == "True":
                    c_val = "v"
                elif cfs[0] == "weka" and cfs[1] == "True":
                    c_algo = "w"
                elif cfs[0] == "maintain" and cfs[1] == "True":
                    flags += "u"
                elif cfs[0] == "reduce_numeric" and cfs[1] == "True":
                    flags += "n"
                elif cfs[0] == "reduce_categoric" and cfs[1] == "True":
                    flags += "o"
                # elif cfs[0] == "slim_opt" and cfs[1] == "True":
                #     flags += "e"
                elif cfs[0] == "use_dt" and cfs[1] == "True":
                    flags += "x"
                elif cfs[0] == "use_encoding_dt" and cfs[1] == "True":
                    flags += "g"
                elif cfs[0] == "size_first" and cfs[1] == "True":
                    flags += "f"
                elif cfs[0] == "recursive" and cfs[1] == "True":
                    flags += "v"
                elif cfs[0] == "incremental" and cfs[1] == "True":
                    flags += "j"

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
