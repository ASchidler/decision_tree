import os
import subprocess
import re

pth = os.path.abspath("./split")
pth_out = os.path.abspath("./trees")
weka_path = os.path.join(os.path.expanduser("~"), "Downloads/weka-3-8-4")
jre_path = os.path.join(weka_path, "jre/zulu11.35.15-ca-fx-jre11.0.5-linux_x64/bin/java")
#-cp ./weka.jar

fls = list(os.listdir(pth))
fls.sort()
for fl in fls:
    if fl.endswith(".data"):
        out_fn = os.path.join(pth_out, fl[:-4]+"tree")
        if not os.path.exists(out_fn):
            print(fl)
            process = subprocess.Popen([
                jre_path, "-cp", os.path.join(weka_path, "weka.jar"),
                "weka.classifiers.trees.J48",
                "-t", os.path.join(pth, fl),
                "-no-cv", "-U", "-J", "-M", "0"],
                cwd=weka_path
                , stdout=subprocess.PIPE)

            output, _ = process.communicate()
            output = output.decode('ascii')

            mt = re.search("J48 unpruned tree[^\-]*[\-]*(.*)Number of Leaves", output, re.DOTALL)
            with open(out_fn, "w") as outp:
                outp.write(mt.group(1).strip())

