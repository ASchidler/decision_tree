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

    for ci, cl in enumerate(fl):
        if type(cl) is not str:
            cl = cl.decode('ascii')

        # First line
        if ci == 0:
            if cl.startswith("Only "):
                # Error if the index is too high
                return
            data_file = cl.split(",")[0].split(":")[1].strip()
            c_flags = cl[cl.find("(")+1:cl.find(")")-1].split(",")
            for cf in c_flags:
                cfs = cf.strip().split("=")

                if cfs[0] == "alt_sat" and cfs[1] == "True":
                    flags += "a"
                elif cfs[0] == "categorical" and cfs[1] == "True":
                    flags += "c"
                elif cfs[0] == "hybrid" and cfs[1] == "True":
                    flags += "y"
                # elif cfs[0] == "size" and cfs[1] == "True":
                #     flags += "z"
                elif cfs[0] == "slice":
                    c_slice = int(cfs[1])
                elif cfs[0] == "use_smt" and cfs[1] == "True":
                    flags += "s"
                elif cfs[0] == "validation" and cfs[1] == "True":
                    flags += "v"

        if done:
            tree_data.append(cl)
        elif cl.startswith("Running depth"):
            depth_lb = int(cl.split(" ")[-1].strip())
        elif cl.startswith("Running size"):
            size_ub = int(cl.split(" ")[-1].strip())
        elif cl.startswith("END Tree"):
            done = True

        out_file = f"{data_file}.{c_slice}.{experiment}.{flags}"
        if len(flags) == 0:
            flags = "0"

        if len(tree_data) > 0:
            out_path = os.path.join("trees", f"{experiment}", out_file+".dt")
            with open(out_path, "w") as out_file:
                out_file.write("".join(tree_data))
        elif depth_lb > 0:
            out_path = os.path.join("trees", f"{experiment}", out_file+".info")
            with open(out_path, "w") as out_file:
                out_file.write(f"{depth_lb}{os.linesep}{size_ub}{os.linesep}")


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
