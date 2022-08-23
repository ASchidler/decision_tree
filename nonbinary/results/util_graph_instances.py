import os
from collections import defaultdict

import nonbinary.nonbinary_instance as inst
import matplotlib.pyplot as plt

instances = defaultdict(lambda: [0, 0])
non_binary_class = 0
no_test_set = 0

ignores = set()
with open("results/ignore.txt") as ig:
    for cig in ig:
        ignores.add(cig.strip())


for c_file in os.listdir("instances"):
    if c_file.endswith(".data"):
        if int(c_file[-6:-5]) == 1:
            instance, test_instance, _ = inst.parse("instances", c_file[:-7], int(c_file[-6:-5]))
            instances[c_file[:-7]][0] += len(instance.examples) + len(test_instance.examples)
            instances[c_file[:-7]][1] = instance.num_features
            if len(instance.classes) > 2:
                non_binary_class += 1

        if int(c_file[-6:-5]) == 5:
            no_test_set += 1
print(f"{non_binary_class} non binary class instances")
fig, ax = plt.subplots(figsize=(4.5, 3), dpi=80)
ax.set_axisbelow(True)

colors = ['black', '#eecc66', '#228833', '#777777', '#004488', '#a50266']
symbols = ['d', 'x', 's', 'o', 'v']

for c_ignore in [True, False]:
    features = [x[1] for inst, x in instances.items() if c_ignore == (inst in ignores)]
    samples = [x[0] for inst, x in instances.items() if c_ignore == (inst in ignores)]
    ax.scatter(samples, features, marker=symbols.pop(), s=30, color=colors.pop())

min_f = min(x[1] for x in instances.values())
min_s = min(x[0] for x in instances.values())
max_f = max(x[1] for x in instances.values())
max_s = max(x[0] for x in instances.values())

print(f"F: {min_f}/{max_f} {min_s}/{max_s}")
print(f"No Test Set: {no_test_set}")

#ax = fig.add_axes([0,0,1,1])

ax.set_ylabel('Features')
ax.set_xlabel('Samples')
#ax.set_title('scatter plot')
#plt.rcParams["legend.loc"] = 'lower left'
plt.rcParams['savefig.pad_inches'] = 0
plt.autoscale(tight=True)
plt.xscale("log")
plt.yscale("log")
plt.xlim(1, max_s + 10000)
plt.ylim(1, max_f + 1000)

plt.plot(color='black', linewidth=0.5, linestyle='dashed')
plt.savefig("instance_data.pdf", bbox_inches='tight')
plt.show()

