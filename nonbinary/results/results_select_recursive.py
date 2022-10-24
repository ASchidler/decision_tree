import os
import nonbinary.tree_parsers as tp

for c_file in sorted(os.listdir("../instances")):
    if c_file.endswith(".data"):
        file_name = c_file[:-5]
        print(file_name)


        trees = []
        validations = []
        for file in os.listdir("trees/w"):
            if file.startswith(file_name) and file.endswith(".dt"):
                validation_file_name = file.split(".")
                validation_file_name[2] = "v"
                validation_file_name[3] = "v" + validation_file_name[3]
                validation_file_name = ".".join(validation_file_name)
                if os.path.exists("trees/v/"+validation_file_name):
                    trees.append(tp.parse_internal_tree("trees/w/" + file))
                    validations.append("trees/v/"+validation_file_name)

        if len(trees) > 0:
            _, _, idx, min_tree = min((t.get_nodes(), t.get_depth(), i, t) for (i, t) in enumerate(trees))

            with open("trees/unpruned/"+file_name+".r.dt", "w") as outp:
                outp.write(min_tree.as_string())
            val_tree = tp.parse_internal_tree(validations[idx])
            with open("trees/validation/"+file_name+".r.dt", "w") as outp:
                outp.write(val_tree.as_string())
