""" Print the time progression from a logfile"""

import sys
import tarfile
import matplotlib.pyplot as plt

logfile = sys.argv[1]
instance_name = sys.argv[2]
slice = sys.argv[3]

c_file = None
entries = []
done = False
with tarfile.open(logfile) as tar_file:
    for ctf in tar_file:
        if done:
            break
        file_parts = ctf.path.split(".")

        if file_parts[-2].startswith("e"):
            # Error file
            continue

        cetf = tar_file.extractfile(ctf)

        for ci, cl in enumerate(cetf):
            if type(cl) is not str:
                cl = cl.decode('ascii')

            if ci == 0:
                if not cl.startswith("Instance: "+instance_name) or cl.find("slice="+ slice) == -1:
                    break
            done = True
            if cl.startswith("START") or cl.startswith("END"):
                cf = cl.split(",")
                cv = [x.strip().split(":")[1].strip() for x in cf]
                entries.append((cv[-1], cv[-5], cv[-4], cv[-3], cv[-2]))
            elif cl.startswith("Time:"):
                # Time: 230.2385	Training 1.0000	Test 0.8406	Depth 019	Nodes 139	Method ma
                cf = cl.split("\t")
                cv = [x.strip().split(" ")[1].strip() for x in cf]
                entries.append((cv[0], cv[-3], cv[-2], cv[1], cv[2]))


x = []
y = []

for ce in entries:
    x.append(float(ce[0]) / 60)
    y.append(float(ce[1]))

fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
ax.set_axisbelow(True)
names = []

ax.set_xlabel('Minutes')
ax.set_ylabel('Size')
# ax.set_title('scatter plot')
plt.rcParams["legend.loc"] = 'upper left'
plt.rcParams['savefig.pad_inches'] = 0
plt.autoscale(tight=True)
#plt.legend(names)
# if c_col == 1:
#     plt.xscale("log")
#     plt.yscale("log")
plt.xlim(min(x)-0.5, max(x)+0.5)
plt.ylim(min(y)-0.5, max(y)+0.5)
#ax.axline([0, 0], [1, 1], linestyle="--", color="grey")
#plt.plot(color='black', linewidth=0.5, linestyle='dashed', markersize=10)
ax.scatter(x, y, marker="o", s=10, color='black')
#pname = "pruned" if use_pruning else "unpruned"
#colname = "d" if c_col == 0 else ("n" if c_col == 1 else "a")
plt.savefig(f"progression_{instance_name}.pdf", bbox_inches='tight')
plt.show()

