#!/usr/bin/env python3
"""Regenerate the pinned const table in src/recursion_consts.rs from `capture_all`'s @@ output.

Usage: gen_recursion_consts.py <capture_log> [<recursion_consts.rs>]

Parses the `@@`-prefixed lines the `capture_all` test prints (all 3 operating points, all layers) and
replaces the region between the `// <<GENERATED CONSTS BEGIN ...>>` / `// <<GENERATED CONSTS END>>`
markers with the three full `const K*_CONSTS: PointConsts` definitions. Run `cargo fmt` afterwards.
"""
import sys
import os

POINTS = ["K500N174", "K1000N348", "K2000N695"]


def parse(log_path):
    pts = {}
    cur = None
    for raw in open(log_path):
        i = raw.find("@@")
        if i < 0:
            continue
        parts = raw[i + 2:].split()
        if not parts:
            continue
        tag, rest = parts[0], parts[1:]
        if tag == "POINT":
            cur = rest[0]
            pts[cur] = {"level1_roots": {}, "fold_roots": {}}
        elif cur is None:
            continue
        elif tag.endswith("_TRACE") or tag == "UNPACKER_NOUT":
            pts[cur][tag] = int(rest[0])
        elif tag.endswith("_COLS"):
            pts[cur][tag] = [(p.rsplit(":", 1)[0], int(p.rsplit(":", 1)[1])) for p in rest]
        elif tag in ("LEAF_ROOT", "UNPACKER_ROOT", "NODE_TARGET", "UNPACKER_PCS"):
            pts[cur][tag] = [int(x) for x in rest]
        elif tag.startswith("LEVEL1_ROOT_"):
            pts[cur]["level1_roots"][int(tag.rsplit("_", 1)[1])] = [int(x) for x in rest]
        elif tag.startswith("FOLD_ROOT_"):
            pts[cur]["fold_roots"][int(tag.rsplit("_", 1)[1])] = [int(x) for x in rest]
    return pts


def root(r):
    assert len(r) == 8, r
    return "[" + ", ".join(str(x) for x in r) + "]"


def cols(c):
    return "&[" + ", ".join(f'("{i}", {s})' for i, s in c) + "]"


def roots7(d):
    return "[" + ", ".join(root(d[a]) for a in range(2, 9)) + "]"


def gen_point(name, p):
    pcs = p["UNPACKER_PCS"]  # pow_bits log_blowup log_last_layer n_queries fold_step lifting
    nt = p["NODE_TARGET"]
    return f"""const {name}_CONSTS: PointConsts = PointConsts {{
    leaf: LayerShape {{ trace_log_size: {p['LEAF_TRACE']}, preprocessed_column_log_sizes: {cols(p['LEAF_COLS'])}, root: {root(p['LEAF_ROOT'])} }},
    level1: NodeLayer {{ trace_log_size: {p['LEVEL1_TRACE']}, preprocessed_column_log_sizes: {cols(p['LEVEL1_COLS'])}, roots: {roots7(p['level1_roots'])} }},
    fold: NodeLayer {{ trace_log_size: {p['FOLD_TRACE']}, preprocessed_column_log_sizes: {cols(p['FOLD_COLS'])}, roots: {roots7(p['fold_roots'])} }},
    node_target: ComponentSizes {{ eq: {nt[0]}, qm31_ops: {nt[1]}, m31_to_u32: {nt[2]}, triple_xor: {nt[3]}, blake_g_gate: {nt[4]} }},
    unpacker: UnpackerConfigConst {{
        pcs: PcsConfig {{ pow_bits: {pcs[0]}, fri_config: FriConfig {{ log_blowup_factor: {pcs[1]}, log_last_layer_degree_bound: {pcs[2]}, n_queries: {pcs[3]}, fold_step: {pcs[4]} }}, lifting_log_size: Some({pcs[5]}) }},
        n_outputs: {p['UNPACKER_NOUT']},
        preprocessed_column_log_sizes: {cols(p['UNPACKER_COLS'])},
        root: {root(p['UNPACKER_ROOT'])},
    }},
}};"""


def main():
    log = sys.argv[1]
    rs = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), "..", "src", "recursion_consts.rs")
    pts = parse(log)
    missing = [n for n in POINTS if n not in pts]
    if missing:
        sys.exit(f"capture log missing points: {missing}")
    body = "\n\n".join(gen_point(n, pts[n]) for n in POINTS)
    src = open(rs).read()
    begin_marker, end_marker = "// <<GENERATED CONSTS BEGIN", "// <<GENERATED CONSTS END>>"
    b, e = src.index(begin_marker), src.index(end_marker)
    b_eol = src.index("\n", src.index("\n", b) + 1) + 1  # keep both BEGIN comment lines
    new = src[:b_eol] + "\n" + body + "\n\n" + src[e:]
    open(rs, "w").write(new)
    print(f"regenerated consts for {POINTS} in {os.path.abspath(rs)}")


main()
