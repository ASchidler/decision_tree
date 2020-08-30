import os
import subprocess
import re
import shutil

pth = os.path.abspath("./split")
pth_out = os.path.abspath("./trees")
pth_iti = "/home/asc/iti/data"
iti_exec = "iti"

fls = list(os.listdir(pth))
fls.sort()
for fl in fls:
    if fl.endswith(".data"):
        ds_name = fl[:-5]
        out_fn = os.path.join(pth_out, ds_name+".iti")

        if not os.path.exists(os.path.join(pth_iti, ds_name)):
            cp_dir = os.path.join(pth_iti, ds_name)
            os.mkdir(cp_dir)
            shutil.copy(os.path.join(pth, ds_name+".names"), os.path.join(cp_dir, "names"))
            shutil.copy(os.path.join(pth, fl), os.path.join(cp_dir, "training.data"))

        #if os.path.exists(out_fn):
        print(fl)
        process = subprocess.Popen([iti_exec, ds_name, "-ltraining", "-qtraining", "-f", "-t", "-w"], stdout=subprocess.PIPE)
        output, _ = process.communicate()
        output = output.decode('ascii')

        tree = False
        tree_str = None
        done = True

        with open(out_fn, "w") as outp:
            for i, l in enumerate(output.split("\n")):
                if tree:
                    if len(l.strip()) > 0:
                        outp.write(f"{l}{os.linesep}")

                if l.startswith("Leaves"):
                    tree = True

