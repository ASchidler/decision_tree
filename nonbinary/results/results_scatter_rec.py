import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import math

cmp_heur = True
use_virtual_best = False
use_virtual_all = False
use_cart = False
use_binoct = False
#experiments = [("k", "SZ,EX"), ("m", "SZ"), ("n", "M"), ("q", "None")]
#experiments = [("k", "DP-SL-SZ"), ("m", "DP-SZ")]
# experiments = [("m", "DP-SZ"), ("k", "DP-SL-SZ"), ("n", "MT-DP"), ("x", "DT Budget"),
#                ("q", "DP"), ("f", "SZ-DP"), ("g", "DT Encoding"), ("r", "Reduce Categoric"),
#                ("h", "2"), ("i", "3")]

experiment = "w"

fields = [(10, "Depth"), (9, "Size"), (12, "Accuracy"), (13, "Avg. Decision Length"),
          (11, "Training Accuracy")
          ]
field_idx = 2
bar_idx = [0, 1, 2]
offset = 66
offsets = [(12, "av"), (30, "avx"), (48, "v"), (66, "vx")]
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

with open(f"results_{experiment}_comp.csv") as inp:
    for i, cl in enumerate(inp):
        if i == 0:
            for c_offset, c_name in offsets:
                if cmp_heur or c_name != offsets[0][1]:
                    legend.append(c_name)
        else:
            for c_offset, c_name in offsets:
                cf = cl.split(";")
                if cmp_heur:
                    if len(results[cf[0]]) == 0:
                        results[cf[0]].append([cf[x[0]] for x in fields])
                    # else:
                    #     if results[cf[0]][0] != cf[target_idx]:
                    #         raise RuntimeError("Base mismatch")
                result1 = [cf[x[0] + c_offset].strip() for x in fields]
                result2 = [cf[x[0] + c_offset + 6].strip() for x in fields]
                results[cf[0]].append(result1)
                # if float(result2[2]) > float(result1[2]):
                #     results[cf[0]].append(result2)
                # else:
                #     results[cf[0]].append(result1)

if use_virtual_best:
    if use_virtual_all:
        for idx in bar_idx:
            legend.append(f"Virtual Best {fields[idx][1]}")
    else:
        legend.append("Virtual Best")

X = [[] for _ in range(0 if cmp_heur else 1, len(legend))]
y = []
lts = [[0 for _ in range(0, len(fields))] for _ in range(0 if cmp_heur else 1, len(legend))]
lts2 = [[0 for _ in range(0, len(fields))] for _ in range(0 if cmp_heur else 1, len(legend))]
gts = [[0 for _ in range(0, len(fields))] for _ in range(0 if cmp_heur else 1, len(legend))]
gts2 = [[0 for _ in range(0, len(fields))] for _ in range(0 if cmp_heur else 1, len(legend))]

for cl in results.values():
    if len(cl) < len(offsets) + (1 if cmp_heur else 0) or any(x == -1 for x in cl):
        continue

    if not any(y == "-1" or y.strip() == "" for x in cl for y in x):
        if use_virtual_best:
            if use_virtual_all:
                for idx in bar_idx:
                    if idx == 2 or idx == 4:
                        max_entry = max((x for x in cl[1:] if x[idx] != "-1"), key=lambda x: (round(float(x[idx]),2), -round(float(x[0]), 2)))
                    else:
                        max_entry = min((x for x in cl[1:] if x[idx] != "-1"), key=lambda x: (round(float(x[idx]),2), -round(float(x[2]),2)))
                    cl.append(max_entry)
            else:
                max_entry = max((x for x in cl[1:] if x[2] != "-1"), key=lambda x: float(x[2]))
                cl.append(max_entry)

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

if use_virtual_best:
    best_counts = {x: 0 for x in range(0, len(legend)-1)}
    for cl in results.values():
        best_val = max(float(x[2]) for x in cl[1:-1 * (len(bar_idx) if use_virtual_all else 1)])

        for i, cv in enumerate(cl[1:-1 * (len(bar_idx) if use_virtual_all else 1)]):
            if math.fabs(best_val - float(cv[2])) <= 0.0001:
                best_counts[i] += 1

    _, best_idx = max((v, k) for k, v in best_counts.items())
    if use_virtual_all:
        legend = ["Single Best", *legend[-1 * len(bar_idx):]]
        X = [X[best_idx], *X[-1 * len(bar_idx):]]
        lts = [lts[best_idx], *lts[-1 * len(bar_idx):]]
        lts2 = [lts2[best_idx], *lts2[-1 * len(bar_idx):]]
        gts = [gts[best_idx], *gts[-1 * len(bar_idx):]]
        gts2 = [gts2[best_idx], *gts2[-1 * len(bar_idx):]]
    else:
        legend = ["Single Best", legend[-1]]
        X = [X[best_idx], X[-1]]
        lts = [lts[best_idx], lts[-1]]
        lts2 = [lts2[best_idx], lts2[-1]]
        gts = [gts[best_idx], gts[-1]]
        gts2 = [gts2[best_idx], gts2[-1]]

# Bar plot
#fig, ax = plt.subplots(figsize=(4, 1 * len(lts)))
yticks = []
ylabels = []

# y_scale = 0.5
# bar_height = 0.09
# # Example data
# for i in range(0, len(lts)):
#     plt.text(0, -i * y_scale, legend[i], fontsize=10, ha='center')
#     yticks.append(-i)
#     ylabels.append(legend[i])
#     plt.plot([0, 0], [-i*y_scale - 0.045, -i*y_scale - 0.045 - len(bar_idx) * (bar_height + 0.02) + 0.027], color='black', lw=0.5)
#     for idx_idx, idx in enumerate(bar_idx):
#         y_pos = -1 * (i * y_scale + (bar_height + 0.02) * idx_idx + 0.1)
#
#         #yticks.append(-1 * (i + 0.15 * (idx_idx + 1)))
#         #ylabels.append(fields[idx][1])
#         # ax.barh(y_pos, gts[i][idx], align='center', color='grey', height=bar_height)
#         # plt.text(gts[i][idx] + 0.1, y_pos - 0.01, str(gts[i][idx]), ha='left', va='center')
#         # ax.barh(y_pos, -lts[i][idx], align='center', color='grey', height=bar_height)
#         # plt.text(-lts[i][idx] - 0.1, y_pos - 0.01, str(lts[i][idx]), ha='right', va='center')
#         # ax.barh(y_pos, gts2[i][idx], align='center', color=colors[3], height=bar_height)
#         # plt.text(0.2, y_pos - 0.01, str(gts2[i][idx]), ha='left', va='center')
#         # ax.barh(y_pos, -lts2[i][idx], align='center', color=colors[3], height=bar_height)
#         # plt.text(-0.2, y_pos - 0.01, str(lts2[i][idx]), ha='right', va='center')
#         ax.barh(y_pos, gts[i][idx], align='center', color='grey', height=bar_height)
#         plt.text(max(gts[i][idx] + 0.1, 3), y_pos - 0.01, f"{gts[i][idx]} ({gts2[i][idx]})", ha='left', va='center', size=8)
#         ax.barh(y_pos, -lts[i][idx], align='center', color='grey', height=bar_height)
#         plt.text(min(-lts[i][idx] - 0.1, -3), y_pos - 0.01, f"{lts[i][idx]} ({lts2[i][idx]})", ha='right', va='center', size=8)
#         ax.barh(y_pos, gts2[i][idx], align='center', color=colors[3], height=bar_height)
#         plt.text(0.2, y_pos - 0.01, fields[idx][1], ha='center', va='center', size=8)
#         ax.barh(y_pos, -lts2[i][idx], align='center', color=colors[3], height=bar_height)
#         #plt.text(-0.2, y_pos - 0.01, str(lts2[i][idx]), ha='right', va='center')
#
# bar_height = 0.06
# y_scale = len(lts) * 2 * bar_height + 0.1
# for idx_idx, idx in enumerate(bar_idx):
#     plt.text(0, -idx_idx * y_scale, fields[idx][1], fontsize=10, ha='left')
#
#     for i in range(0, len(lts)):
#         #plt.plot([0, 0], [-i*y_scale - 0.045, -i*y_scale - 0.045 - len(bar_idx) * (bar_height + 0.02) + 0.027], color='black', lw=0.5)
#         y_pos = -1 * (idx_idx * y_scale + 2 * i * (bar_height + 0.01) + 0.04)
#
#         plt.text(0.2, y_pos, f"{legend[i]} >", ha='left', va='center', size=8) #  {'C4.5' if cmp_heur else experiments[0][1]}
#         plt.text(0.2, y_pos - bar_height, f"{legend[i]} <", ha='left', va='center', size=8)
#
#         ax.barh(y_pos, gts[i][idx], align='center', color='grey', height=bar_height)
#         ax.barh(y_pos, gts2[i][idx], align='center', color=colors[3 if idx != 2 else 0], height=bar_height)
#         plt.text(max(gts[i][idx] + 0.1, 3), y_pos - 0.01, f"{gts[i][idx]} ({gts2[i][idx]})", ha='left', va='center', size=8)
#
#         ax.barh(y_pos - bar_height, lts[i][idx], align='center', color='grey', height=bar_height)
#         ax.barh(y_pos - bar_height, lts2[i][idx], align='center', color=colors[0 if idx != 2 else 3], height=bar_height)
#         plt.text(max(lts[i][idx] + 0.1, 3), y_pos - bar_height - 0.01, f"{lts[i][idx]} ({lts2[i][idx]})", ha='left', va='center', size=8)
#

#lts = [lts[0]]
fig, ax = plt.subplots(figsize=(1.8, 0.75 * len(lts)))
bar_height = 0.025
y_scale = len(lts) * bar_height + 0.04
for idx_idx, idx in enumerate(bar_idx):
    plt.text(len(results)/2, -idx_idx * y_scale - 0.003, fields[idx][1], fontsize=10, ha='center', color='black')

    for i in range(0, len(lts)):
        #plt.plot([0, 0], [-i*y_scale - 0.045, -i*y_scale - 0.045 - len(bar_idx) * (bar_height + 0.02) + 0.027], color='black', lw=0.5)
        y_pos = -1 * (idx_idx * y_scale + i * (bar_height + 0.005) + 0.02)

        plt.text(0+0.2, y_pos, f"{legend[i]}", ha='left', va='center', size=8, color="black", fontweight=600)
        # plt.text(0.2, y_pos - bar_height, f"{legend[i]} <", ha='left', va='center', size=8)
        ax.barh(y_pos, lts2[i][idx], align='center', color='#4eb265' if idx != 2 else '#d6604d', height=bar_height,
                label=lts2[i][idx])
        ax.barh(y_pos, lts[i][idx] - lts2[i][idx], align='center', color='#cae0ab' if idx != 2 else '#F5a582',
                height=bar_height, left=lts2[i][idx])

        ax.barh(y_pos, len(results) - gts[i][idx] - lts[i][idx], left=lts[i][idx], align='center', color='#FFEE99',
                height=bar_height)

        ax.barh(y_pos, gts[i][idx] - gts2[i][idx], align='center', color='#F5a582' if idx != 2 else '#cae0ab',
                height=bar_height, left=len(results) - gts[i][idx])
        ax.barh(y_pos, gts2[i][idx], align='center', color='#d6604d' if idx != 2 else '#4eb265', height=bar_height,
                left=len(results) - gts2[i][idx])


        #plt.text(max(gts[i][idx] + 0.1, 3), y_pos - 0.01, f"{gts[i][idx]} ({gts2[i][idx]})", ha='left', va='center',
        #         size=8)
        #plt.text(max(lts[i][idx] + 0.1, 3), y_pos - bar_height - 0.01, f"{lts[i][idx]} ({lts2[i][idx]})", ha='left', va='center', size=8)




# people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
# y_pos = np.arange(len(people))
# performance = 3 + 10 * np.random.rand(len(people))
# error = np.random.rand(len(people))
#fig.axes[0].get_xaxis().set_visible(False)
ax.set_yticks([])
ax.set_yticklabels([])
ax.spines['left'].set_color('none')
#set_position('zero')
#ax.spines['bottom'].set_color('none')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

#ax.invert_yaxis()  # labels read top-to-bottom
#ax.set_xlabel('Instances')
plt.rcParams['savefig.pad_inches'] = 0
plt.autoscale(tight=True)

plt.savefig(f"bar_prune.pdf", bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(4, 2.5), dpi=300)
max_y = max(max_y, max(y))
min_y = min(min_y, min(y))

for x in X:
    max_x = max(max_x, max(x))
    min_x = min(min_x, min(x))

    ax.scatter(x, y, marker=symbols.pop(), s=10, alpha=0.7 if colors[-1] != '#eecc66' else 1,
               zorder=2 if colors[-1] != '#eecc66' else 1, color=colors.pop())

ax.set_axisbelow(True)
names = []

ax.set_xlabel('DT-SLIM')
ax.set_ylabel('C4.5' if cmp_heur else offsets[0][1])
# ax.set_title('scatter plot')
plt.rcParams["legend.loc"] = 'upper left'
plt.rcParams['savefig.pad_inches'] = 0
plt.autoscale(tight=True)
plt.legend([x.replace("Virtual ","") for x in legend])
if field_idx == 1:
    plt.xscale("log")
    plt.yscale("log")
if field_idx == 0 or field_idx == 3:
    plt.xlim(min_x - 1, max_x + 1)
    plt.ylim(min_y - 1, max_y + 1)
if field_idx == 1:
    plt.xlim(min_x, max_x+200)
    plt.ylim(min_y, max_y+200)
if field_idx == 2:
    plt.xlim(min_x-0.01, max_x+0.01)
    plt.ylim(min_y-0.01, max_y+0.01)
ax.axline([0, 0], [1, 1], linestyle="--", color="grey", zorder=0)

plt.savefig(f"scatter_prune_{field_idx}.pdf", bbox_inches='tight')
plt.show()
