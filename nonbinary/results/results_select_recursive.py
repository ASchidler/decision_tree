import os
import nonbinary.tree_parsers as tp

for c_file in sorted(os.listdir("../instances")):
    if c_file.endswith(".data"):
        file_name = c_file[:-5]
        print(file_name)

        trees = []
        for file in os.listdir("trees/w"):
            if file.startswith(file_name) and file.endswith(".dt"):
                trees.append(tp.parse_internal_tree("trees/w/"+file))

        if len(trees) > 0:
            _, _, _, min_tree = min((t.get_nodes(), t.get_depth(), i, t) for (i, t) in enumerate(trees))

            with open("trees/unpruned/"+file_name+".r.dt", "w") as outp:
                outp.write(min_tree.as_string())

