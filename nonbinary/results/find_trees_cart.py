import os
import subprocess
import re
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
import parser
import decision_tree as dt
from collections import defaultdict

validate_path = os.path.abspath("./validate")
use_validation = True
main_path = os.path.abspath("./split")
pth = os.path.abspath("./split") if not use_validation else os.path.abspath("./validate")
pth_out = os.path.abspath("./trees") if not use_validation else os.path.abspath("./validate-trees")

pruning = 1  # 0 is no pruning

fls = [x for x in os.listdir(pth) if x.endswith("data")]
fls.sort()
for fl in fls[5::6]:
    if fl.endswith("data"):
        if pruning == 0:
            out_fn = os.path.join(pth_out, fl[:-4]+"cart")
        else:
            out_fn = os.path.join(pth_out, fl[:-4] + f"p{pruning}." + "cart")

        if not os.path.exists(out_fn):
            print(out_fn)
            # Same file
            training_instance = parser.parse(os.path.join(pth, fl), has_header=False) if not use_validation else \
                parser.parse(os.path.join(main_path, fl.split(".")[0] + "." + fl.split(".")[-1]), has_header=False)
            test_instance = parser.parse(os.path.join(pth, fl[:-4] + "test"), has_header=False)

            x = [[f for f in k.features if f is not None] for k in training_instance.examples]
            y = [k.cls for k in training_instance.examples]
            test_x = [[f for f in k.features if f is not None] for k in test_instance.examples]
            test_y = [k.cls for k in test_instance.examples]

            is_tf = any(isinstance(k, bool) for k in training_instance.examples[1].features)
            cls = tree.DecisionTreeClassifier()
            cls.fit(x, y)
            if pruning == 1:
                path = cls.cost_complexity_pruning_path(x, y)
                ccp_alphas, impurities = path.ccp_alphas, path.impurities
                alpha_precision = defaultdict(int)

                if not use_validation:
                    for c_train, c_test in list(StratifiedKFold().split(x, y=y)):
                        c_x = [x[i] for i in c_train]
                        c_y = [y[i] for i in c_train]
                        t_x = [x[i] for i in c_test]
                        t_y = [y[i] for i in c_test]

                        for c_alpha in ccp_alphas:
                            cls_tmp = tree.DecisionTreeClassifier(ccp_alpha=c_alpha)
                            cls_tmp.fit(c_x, c_y)
                            alpha_precision[c_alpha] += cls_tmp.score(t_x, t_y)
                else:
                    validation_training = parser.parse(os.path.join(pth, fl), has_header=False)
                    train_x = [[f for f in k.features if f is not None] for k in validation_training.examples]
                    train_y = [k.cls for k in validation_training.examples]
                    validation_instance = parser.parse(os.path.join(pth, fl[:-4] + "validate"), has_header=False)
                    valid_x = [[f for f in k.features if f is not None] for k in validation_instance.examples]
                    valid_y = [k.cls for k in validation_instance.examples]
                    for c_alpha in ccp_alphas:
                        cls_tmp = tree.DecisionTreeClassifier(ccp_alpha=c_alpha)
                        cls_tmp.fit(train_x, train_y)
                        alpha_precision[c_alpha] += cls_tmp.score(valid_x, valid_y)

                best_alpha, _ = max(alpha_precision.items(), key=lambda k: k[1])
                cls = tree.DecisionTreeClassifier(ccp_alpha=best_alpha)
                cls.fit(x, y)

            # Convert
            print(f"Computed {fl}, accuracy: {cls.score(x, y)}/{cls.score(test_x, test_y)}")
            new_tree = dt.DecisionTree(training_instance.num_features, cls.tree_.capacity)
            classes = list(cls.classes_)

            for i in range(0, len(cls.tree_.feature)):
                if cls.tree_.feature[i] >= 0:
                    new_tree.nodes[i+1] = dt.DecisionTreeNode(cls.tree_.feature[i]+1, i+1)
                else:
                    c_max = (-1, None)
                    for cc in range(0, len(classes)):
                        c_max = max(c_max, (cls.tree_.value[i][0][cc], cc))
                    new_tree.nodes[i+1] = dt.DecisionTreeLeaf(classes[c_max[1]].strip(), i+1)

            for i in range(0, cls.tree_.capacity):
                if cls.tree_.children_left[i] > 0:
                    new_tree.nodes[i + 1].right = new_tree.nodes[cls.tree_.children_left[i]+1]
                    new_tree.nodes[i + 1].left = new_tree.nodes[cls.tree_.children_right[i] + 1]
            new_tree.root = new_tree.nodes[1]

            with open(out_fn, "w") as outf:
                outf.write(new_tree.as_string())

            # new_tree.root = new_tree.nodes[1]
            # print(f"Accuracy: {new_tree.get_accuracy(training_instance.examples)}")
