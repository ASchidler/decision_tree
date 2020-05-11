import base_encoding
from decision_tree import DecisionDiagram


class DiagramDepthEncoding(base_encoding.BaseEncoding):
    def __init__(self, stream):
        base_encoding.BaseEncoding.__init__(self, stream)
        self.g = None
        self.d = None

    def init_vars(self, instance, depth):
        self.d = []
        for i in range(0, len(instance.examples)):
            self.d.append([])
            for dl in range(0, depth):
                self.d[i].append([None])
                for f in range(1, instance.num_features + 1):
                    self.d[i][dl].append(self.add_var())

        self.g = [{} for _ in range(0, len(instance.examples))]
        for i in range(0, len(instance.examples)):
            for j in range(i + 1, len(instance.examples)):
                self.g[i][j] = [self.add_var() for _ in range(0, depth + 1)]

    def encode(self, instance, depth):
        self.init_vars(instance, depth)

        # Add level 0, all examples are in the same group
        for i in range(0, len(instance.examples)):
            for j in range(i + 1, len(instance.examples)):
                self.add_clause(self.g[i][j][0])

        # Verify that at last level, the partitioning is by class
        for i in range(0, len(instance.examples)):
            for j in range(i + 1, len(instance.examples)):
                if instance.examples[i].cls == instance.examples[j].cls:
                    self.add_clause(self.g[i][j][depth])
                else:
                    self.add_clause(-self.g[i][j][depth])

        # Transitivity of group membership
        for i in range(0, len(instance.examples)):
            for j in range(i + 1, len(instance.examples)):
                for dl in range(1, depth):
                    for k in range(j + 1, len(instance.examples)):
                        if i != k and j != k:
                            self.add_clause(-self.g[i][j][dl], -self.g[j][k][dl], self.g[i][k][dl])
                            self.add_clause(-self.g[i][j][dl], -self.g[i][k][dl], self.g[j][k][dl])
                            self.add_clause(-self.g[i][k][dl], -self.g[j][k][dl], self.g[i][j][dl])

        # Verify that the examples are partitioned correctly
        for i in range(0, len(instance.examples)):
            for j in range(i + 1, len(instance.examples)):
                for dl in range(0, depth):
                    for f in range(1, instance.num_features+1):
                        if instance.examples[i].features[f] == instance.examples[j].features[f]:
                            self.add_clause(-self.g[i][j][dl], -self.d[i][dl][f], self.g[i][j][dl+1])
                        else:
                            self.add_clause(-self.g[i][j][dl], -self.d[i][dl][f], -self.g[i][j][dl+1])

        # Verify that d is consistent
        for i in range(0, len(instance.examples)):
            for j in range(i + 1, len(instance.examples)):
                for dl in range(0, depth):
                    for f in range(1, instance.num_features+1):
                        self.add_clause(-self.g[i][j][dl], -self.d[i][dl][f], self.d[j][dl][f])

        # One feature per level and group
        for i in range(0, len(instance.examples)):
            for dl in range(0, depth):
                clause = []
                for f in range(1, instance.num_features + 1):
                    clause.append(self.d[i][dl][f])
                    for f2 in range(f + 1, instance.num_features + 1):
                        self.add_clause(-self.d[i][dl][f], -self.d[i][dl][f2])
                self.add_clause(*clause)

        self.write_header(instance)

    def decode2(self, model, instance, depth):
        tree = DecisionDiagram(instance.num_features, 2**(depth+1) - 1)

        def find_feature(ce, cdl):
            ce_feature = None
            for cf in range(1, instance.num_features+1):
                if model[self.d[ce][cdl][cf]]:
                    if ce_feature is None:
                        ce_feature = cf
                    else:
                        print(f"ERROR double feature {cf} and {ce_feature} for experiment {ce}, at level {cdl}.")
            if ce_feature is None:
                print(f"ERROR no feature for {ce} at level {cdl}.")
            return ce_feature

        # Start with pos, neg groups
        node_id = 2**(depth+1) - 1
        dummy = 2**(depth+1)
        groups = {
            node_id: ([i for i in range(0, len(instance.examples)) if not instance.examples[i].cls], None, None, None),
            node_id-1: ([i for i in range(0, len(instance.examples)) if instance.examples[i].cls], None, None, None)
        }
        node_id -= 2

        # Parse bottom up. We already have the two leafs, now find inner nodes
        for dl in range(depth - 1, -1, -1):
            new_groups = {}
            # Each group resembles a node
            while groups:
                p = next(iter(groups))
                g, lc, rc, feat = groups.pop(p)
                # Ensure ordering within the group. Avoids smaller/larger checks
                g.sort(reverse=True)

                # Not a leaf
                if lc is not None or rc is not None:
                    # If only on child, we can omit the node
                    if lc is None and depth > 0:
                        p = rc
                    elif rc is None and depth > 0:
                        p = lc
                    else:
                        # This is an edge case happening if all examples have the same class -> we have a depth 1 tree
                        # Connect the root to the unused leaf
                        if rc is None:
                            rc = dummy - 1 if instance.examples[0].cls else dummy - 2
                        if lc is None:
                            lc = dummy - 1 if instance.examples[0].cls else dummy - 2
                        # edge case end

                        tree.add_node(p, feat, lc, rc)
                        tree.root = tree.nodes[p]  # Root is the last added node

                # Separate g into groups
                while g:
                    n = g.pop()
                    found = False
                    feat = find_feature(n, dl)
                    feat_val = instance.examples[n].features[feat]

                    for c_id, (c_g, c_lc, c_rc, c_ft) in new_groups.items():
                        n1 = c_g[0] if c_g[0] < n else n
                        n2 = n if c_g[0] < n else c_g[0]
                        if model[self.g[n1][n2][dl]]:
                            if c_ft != feat:
                                print("ERROR: Two items belonging to the same group are split on a different feature")
                            if feat_val:
                                if c_lc is None:
                                    new_groups[c_id] = (c_g, p, c_rc, c_ft)
                                elif c_lc != p:
                                    print("ERROR: Left child already set, but is a different left child")
                            else:
                                if c_rc is None:
                                    new_groups[c_id] = (c_g, c_lc, p, c_ft)
                                elif c_rc != p:
                                    print("ERROR: Right child already set, but is a different right child")
                            c_g.append(n)

                            found = True
                            break

                    if not found:
                        if feat_val:
                            new_groups[node_id] = ([n], p, None, feat)
                        else:
                            new_groups[node_id] = ([n], None, p, feat)
                        node_id -= 1

            groups = new_groups
            for i in range(0, len(instance.examples)):
                cnt = 0
                for g, _, _, _ in groups.values():
                    if i in g:
                        cnt += 1
                if cnt != 1:
                    print("ERROR: Each example should occur in exactly one group")
            for g, _, _, ft in new_groups.values():
                g.sort()
                gs = dict()
                for i in g:
                    found = False
                    for nm, nmv in gs.items():
                        if model[self.g[nm][i][dl + 1]]:
                            found = True
                            nmv.append(i)
                            break
                    if not found:
                        gs[i] = [i]
                    for j in g:
                        if i < j:
                            if not model[self.g[i][j][dl]]:
                                print("ERROR examples that are not in the same group are")
                if len(gs) > 2:
                    print(f"ERROR each group should be partitioned into 1-2 groups, not {len(groups)}")

                for k, v in gs.items():
                    for i in v:
                        for j in v:
                            if instance.examples[i].features[ft] != instance.examples[j].features[ft]:
                                print("ERROR: Examples in group have different feature value")
                            if dl == depth - 1 and instance.examples[i].cls != instance.examples[j].cls:
                                print("ERROR: Leaf has different truth values")

                        for k2, v2 in gs.items():
                            if k != k2:
                                for j in v2:
                                    if instance.examples[i].features[ft] == instance.examples[j].features[ft]:
                                        print("ERROR: Examples in split group have same feature value")
                                    if dl == depth - 1 and instance.examples[i].cls == instance.examples[j].cls:
                                        print("ERROR: Split leaf has same truth value")

        return tree

    def decode(self, model, instance, depth):
        def find_feature(ce, cdl):
            ce_feature = None
            for cf in range(1, instance.num_features + 1):
                if model[self.d[ce][cdl][cf]]:
                    if ce_feature is None:
                        ce_feature = cf
                    else:
                        print(f"ERROR double feature {cf} and {ce_feature} for experiment {ce}, at level {cdl}.")
            if ce_feature is None:
                print(f"ERROR no feature for {ce} at level {cdl}.")
            return ce_feature

        nodes = [None, None]
        groups = {1: (0, [i for i in range(0, len(instance.examples))])}
        node_id = 2

        for dl in range(1, depth+1):
            new_groups = {}
            for nid, (_, g) in groups.items():
                c_ft = find_feature(g[0], dl-1)

                g.sort(reverse=True)
                n = g.pop()
                g1 = [n]
                g2 = []
                lc = None
                rc = None

                while g:
                    n2 = g.pop()
                    if model[self.g[n][n2][dl]]:
                        g1.append(n2)
                    else:
                        g2.append(n2)

                for cg in [g1, g2]:
                    if len(cg) > 0:
                        found = False
                        cn = cg[0]
                        new_id = -1
                        for nid2, (rep, og) in new_groups.items():
                            cn1 = min(cn, rep)
                            cn2 = max(cn, rep)
                            if model[self.g[cn1][cn2][dl]]:
                                og.extend(cg)
                                found = True
                                new_id = nid2
                                break
                        if not found:
                            new_groups[node_id] = (cn, cg)
                            new_id = node_id
                            node_id += 1
                            nodes.append(None)

                        if instance.examples[cn].features[c_ft]:
                            lc = new_id
                        else:
                            rc = new_id
                nodes[nid] = (lc, rc, c_ft)
            groups = new_groups

        tree = DecisionDiagram(instance.num_features, len(nodes)-1)
        dummy = len(nodes)

        # Depending on which of the two groups contains the true samples, we may have to switch the leaf values
        cls_key = min(groups.keys())
        if not instance.examples[groups[cls_key][0]].cls:
            tree.nodes[len(nodes)-2].cls = False
            tree.nodes[len(nodes)-1].cls = True

        for i in range(len(nodes)-3, 0, -1):# -3, -2 for the leafs and -1 to start at last index. Also 0 is always None
            tree.add_node(i, nodes[i][2], nodes[i][0] if nodes[i][0] is not None else dummy, nodes[i][1] if nodes[i][1] is not None else dummy)

        tree.root = tree.nodes[1]

        return tree

    def check_consistency(self, model, instance, num_nodes, tree):
        pass

    @staticmethod
    def new_bound(tree, instance):
        if tree is None:
            return 1

        def dfs_find(node, level):
            if node.is_leaf:
                return level
            else:
                return max(dfs_find(node.left, level + 1), dfs_find(node.right, level + 1))

        return dfs_find(tree.root, 0)

    @staticmethod
    def lb():
        return 1
