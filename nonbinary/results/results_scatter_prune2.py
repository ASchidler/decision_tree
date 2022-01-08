import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import math

cmp_heur = True
use_cart = False
use_binoct = False

experiments = [("m", "DT-SLIM")]

#experiments = [("w", "Recursive")]

all_experiments = ["k", "m"]

fields = [(10, "Depth"), (9, "Size"), (12, "Accuracy"), (13, "Avg. Decision Length"),
          (11, "Training Accuracy")
          ]
field_idx = 1
bar_idx = [0, 1, 2]
offset = 12
#offset = 66
#offset = 48

colors = ['#228833', 'black', '#eecc66', '#bb5566', '#004488']
symbols = ['d', 'x', 's', 'v', 'o']


min_x = sys.maxsize * 1.0
max_x = 0.0
min_y = sys.maxsize * 1.0
max_y = 0.0

results = defaultdict(list)
legend = []

target_idx = fields[field_idx][0]

for c_experiment, c_ex_name in experiments:
    legend.append(c_ex_name)

    with open(f"results_{c_experiment}_comp{('' if c_experiment != 'm' or not use_binoct else '_binoct') if not use_cart else '_c'}.csv") as inp:
        for i, cl in enumerate(inp):
            if i > 0:
                cf = cl.split(";")
                if cmp_heur:
                    if len(results[cf[0]]) == 0:
                        results[cf[0]].append([cf[x[0]] for x in fields])
                    # else:
                    #     if results[cf[0]][0] != cf[target_idx]:
                    #         raise RuntimeError("Base mismatch")
                else:
                    result1 = [cf[x[0] + offset - 6].strip() for x in fields]
                    results[cf[0]].append(result1)
                result1 = [cf[x[0] + offset].strip() for x in fields]
                result2 = [cf[x[0] + offset + 6].strip() for x in fields]
                results[cf[0]].append(result1)
                # if float(result2[2]) > float(result1[2]):
                #     results[cf[0]].append(result2)
                # else:
                #     results[cf[0]].append(result1)

X = [[] for _ in range(0, len(legend))]
y = []
lts = [[0 for _ in range(0, len(fields))] for _ in range(0, len(legend))]
lts2 = [[0 for _ in range(0, len(fields))] for _ in range(0, len(legend))]
gts = [[0 for _ in range(0, len(fields))] for _ in range(0, len(legend))]
gts2 = [[0 for _ in range(0, len(fields))] for _ in range(0, len(legend))]

for k, cl in results.items():
    if len(cl) < len(experiments) + 1 or any(x == -1 for x in cl):
        continue

    if not any(z == "-1" or z.strip() == "" for x in cl for z in x):
        y.append(float(cl[0][field_idx]))
        for i, cv in enumerate(cl[1:]):
            X[i].append(float(cv[field_idx]))
            for idx in range(0, len(fields)):
                if round(float(cv[idx]), 2) < round(float(cl[0][idx]), 2):
                    lts[i][idx] += 1
                    if (idx != 2 and float(cv[idx]) < float(cl[0][idx]) * 0.9) or \
                        (idx == 2 and float(cl[0][idx]) - float(cv[idx]) > 0.01):
                        lts2[i][idx] += 1
                elif round(float(cv[idx]), 2) > round(float(cl[0][idx]),2):
                    gts[i][idx] += 1
                    if (idx != 2 and float(cv[idx]) * 0.9 > float(cl[0][idx])) or \
                            (idx == 2 and float(cv[idx]) - float(cl[0][idx]) > 0.01):
                        gts2[i][idx] += 1

for i in range(0, len(lts)):
    print(f"{i}: {lts[i]} ({lts2[i]}) {gts[i]} ({gts2[i]})")

fig, ax = plt.subplots(figsize=(4, 2.5), dpi=300)
max_y = max(max_y, max(y))
min_y = min(min_y, min(y))

# for x in X:
#     max_x = max(max_x, max(x))
#     min_x = min(min_x, min(x))
#
#     ax.scatter(x, y, marker=symbols.pop(), s=10, alpha=0.7 if colors[-1] != '#eecc66' else 1,
#                zorder=2 if colors[-1] != '#eecc66' else 1, color=colors.pop())

for x in X:
    max_x = max(max_x, max(x))
    min_x = min(min_x, min(x))

    if field_idx == 2:
        xp = [round(x[i] - y[i], 2) for i in range(0, len(x))]
    else:
        xp = [round((x[i] - y[i]) / y[i], 2) for i in range(0, len(x))]
        xp = [round((x[i] / y[i] * 100), 2) for i in range(0, len(x))]
    #xp = [v if v != 0 else 0.00000001 for v in xp]
    xp.sort()

    ax.scatter(range(0, len(x)), xp, marker=symbols.pop(), s=3, alpha=0.7 if colors[-1] != '#eecc66' else 1,
               zorder=2 if colors[-1] != '#eecc66' else 1, color=colors.pop())

    for c_idx, c_v in enumerate(xp):
        if c_v > 100:
            if c_idx > 0:
                ax.axvline(c_idx - 0.5, linestyle="--", color="grey", zorder=0, linewidth=0.5)
            break

ax.set_axisbelow(True)
names = []

#ax.axline([0, 0], [1, 1], linestyle="--", color="grey", zorder=0)

ax.set_xlabel('Instances')
ax.set_ylabel('Difference')
# ax.set_title('scatter plot')
plt.rcParams["legend.loc"] = 'upper left'
plt.rcParams['savefig.pad_inches'] = 0
plt.autoscale(tight=True)
#plt.legend([x.replace("Virtual ","") for x in legend])
# if field_idx == 1:
#     #plt.xscale("log")
#     plt.yscale("symlog")
# if field_idx == 0 or field_idx == 3:
#     plt.xlim(min_x - 1, max_x + 1)
#     plt.ylim(min_y - 1, max_y + 1)
# if field_idx == 1:
#     plt.xlim(min_x, max_x+200)
#     plt.ylim(min_y, max_y+200)
# if field_idx == 2:
#     plt.xlim(min_x-0.01, max_x+0.01)
#     plt.ylim(min_y-0.01, max_y+0.01)
# ax.axline([0, 0], [1, 1], linestyle="--", color="grey", zorder=0)

plt.axhline(linestyle="-", color="grey", zorder=0, linewidth=0.5, y = 100)

plt.savefig(f"scatter_prune_{field_idx}.pdf", bbox_inches='tight')
plt.show()
