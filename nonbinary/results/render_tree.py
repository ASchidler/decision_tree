import os
import subprocess

import nonbinary.tree_parsers as tp

instance_name = "shuttleM"

def dot_export(ct):
    q = [ct.root]

    output = "strict digraph dt {node [label=\"\",width=0.1,height=0.1,fixedsize=true,style=filled];" + os.linesep

    while q:
        c_n = q.pop()
        if c_n.is_leaf:
            continue

        q.extend([c_n.left, c_n.right])
        output += f"n{c_n.id} -> n{c_n.left.id};" + os.linesep
        output += f"n{c_n.id} -> n{c_n.right.id};" + os.linesep

    output += "}" + os.linesep

    return output

#tree1 = f"results/trees/z/{instance_name}.1.z.61z.w.dt"
tree1 = f"results/trees/e/{instance_name}.1.e.70z.e.dt"
tree2 = f"results/trees/unpruned/{instance_name}.1.w.dt"

tree1 = tp.parse_internal_tree(tree1)
tree2 = tp.parse_internal_tree(tree2)

for c_tree, c_name in [(tree1, f"tree_{instance_name}_slim"), (tree2, f"tree_{instance_name}_heuristic")]:
    outp = dot_export(c_tree)
    with open(f"{c_name}.dot", "w") as f:
        f.write(outp)
    with open(f"{c_name}.png", "w") as f:
        subprocess.run(["dot", "-Tpng", f"{c_name}.dot"], stdout=f)

