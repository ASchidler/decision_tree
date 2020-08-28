import time
import sys
import improve.improve_depth_first as df
import improve.size_improver as si
import improve.improver as imp

sample_limit = 200
size_limit = 15


def run(tree, instance, test, tmp_dir="."):
    start_time = time.time()

    def process_change(mth):
        print(f"Time: {time.time() - start_time:.4f}\t"
              f"Training {tree.get_accuracy(instance.examples):.4f}\t"
              f"Test {tree.get_accuracy(test.examples):.4f}\t"
              f"Depth {tree.get_depth():03}\t"
              f"Avg {tree.get_avg_depth():03.4f}\t"
              f"Nodes {tree.get_nodes()}\t"
              f"Method {mth}")
        sys.stdout.flush()
        # for c_n in c_path:
        #     ignore.discard(c_n.id)

    assigned = df.assign_samples(tree, instance)

    done = set()


    def structure(root):
        c_leafs = []
        c_q = [root]
        c_p = {root.id: None}

        while c_q:
            c_n = c_q.pop()
            if c_n.is_leaf:
                c_leafs.append(c_n)
            else:
                c_p[c_n.left.id] = c_n.id
                c_p[c_n.right.id] = c_n.id
                c_q.append(c_n.left)
                c_q.append(c_n.right)

        return c_p, c_leafs

    p, leafs = structure(tree.root)
    leafs.sort(key=lambda x: imp.depth_from(x), reverse=True)

    for c_l in leafs:
        if c_l.id not in done and tree.nodes[c_l.id] is not None:
            path = []
            c_n = c_l.id
            while c_n is not None:
                path.append(tree.nodes[c_n])
                c_n = p[c_n]

            changed = False
            for i in range(len(path)-1, -1, -1):
                if si.mid_reduce(tree, instance, i, path, assigned, size_limit, sample_limit, False, tmp_dir=tmp_dir)[0]:
                    process_change("mr")
                    assigned = df.assign_samples(tree, instance)
                    p, _ = structure(tree.root)
                    changed = True
                    break
            if changed:
                continue

            r, idx = si.leaf_select(tree, instance, 0, path, assigned, size_limit, sample_limit, tmp_dir=tmp_dir)

            if r:
                process_change("ls")
                assigned = df.assign_samples(tree, instance)
                p, _ = structure(tree.root)

            r2, idx2 = si.leaf_reduce(tree, instance, idx+1, path, assigned, size_limit, sample_limit, False, tmp_dir=tmp_dir)
            if r2:
                process_change("la")
                assigned = df.assign_samples(tree, instance)
                p, _ = structure(tree.root)
            r3, idx2 = si.leaf_reduce(tree, instance, idx2+1, path, assigned, size_limit, sample_limit, True,
                                      tmp_dir=tmp_dir)
            if r3:
                process_change("lr")
                assigned = df.assign_samples(tree, instance)
                p, _ = structure(tree.root)

            if not (r or r2 or r3):
                q = [path[idx]]
                while q:
                    c_n = q.pop()
                    done.add(c_n.id)

                    if not c_n.is_leaf:
                        q.append(c_n.left)
                        q.append(c_n.right)
