#!/usr/bin/env python3
"""Regenerate the pinned RECURSION_CONFIG const in src/recursion_consts.rs from `capture_all`'s @@ output.

Usage: gen_recursion_consts.py <capture_log> [<recursion_consts.rs>]

Parses the `@@`-prefixed lines a single-point `capture_all` run prints (one `@@POINT K<k>N<n>` plus its
per-layer lines) and replaces the region between the `// <<GENERATED CONSTS BEGIN ...>>` /
`// <<GENERATED CONSTS END>>` markers with the single `pub const RECURSION_CONFIG: RecursionConfig`
definition (the config type from `recursive_aggregate::pinned_configs`) plus the captured
`CAPTURED_K` / `CAPTURED_SHOTS` / `CAPTURED_N` params. The recursion params are not captured: the
blowups/arity reference the `RECURSION_LOG_BLOWUP` / `FOLD_ARITY` consts and `n_leaves` is read from the
point name (`K<k>N<n>`). `shots` is recomputed from `k` via the `floor(2^26 / (k*N_GATES))` formula.
Run `cargo fmt` afterwards.
"""
import sys
import os

N_GATES = 2547  # iadd256 gate count (matches recursion_consts::N_GATES)


def parse(log_path):
    pt = None
    name = None
    for raw in open(log_path):
        i = raw.find("@@")
        if i < 0:
            continue
        parts = raw[i + 2:].split()
        if not parts:
            continue
        tag, rest = parts[0], parts[1:]
        if tag == "POINT":
            name = rest[0]
            pt = {"level1_roots": {}, "fold_roots": {}}
        elif pt is None:
            continue
        elif tag.endswith("_TRACE") or tag == "UNPACKER_NOUT":
            pt[tag] = int(rest[0])
        elif tag.endswith("_COLS"):
            pt[tag] = [(p.rsplit(":", 1)[0], int(p.rsplit(":", 1)[1])) for p in rest]
        elif tag in ("LEAF_ROOT", "UNPACKER_ROOT", "NODE_TARGET", "UNPACKER_PCS"):
            pt[tag] = [int(x) for x in rest]
        elif tag.startswith("LEVEL1_ROOT_"):
            pt["level1_roots"][int(tag.rsplit("_", 1)[1])] = [int(x) for x in rest]
        elif tag.startswith("FOLD_ROOT_"):
            pt["fold_roots"][int(tag.rsplit("_", 1)[1])] = [int(x) for x in rest]
    return name, pt


def root(r):
    assert len(r) == 8, r
    return "[" + ", ".join(str(x) for x in r) + "]"


def cols(c):
    return "&[" + ", ".join(f'("{i}", {s})' for i, s in c) + "]"


def roots7(d):
    return "&[" + ", ".join(root(d[a]) for a in range(2, 9)) + "]"


def gen_config(p):
    pcs = p["UNPACKER_PCS"]  # pow_bits log_blowup log_last_layer n_queries fold_step lifting
    nt = p["NODE_TARGET"]
    return f"""pub const RECURSION_CONFIG: RecursionConfig = RecursionConfig {{
    leaf: PinnedLayer {{ trace_log_size: {p['LEAF_TRACE']}, preprocessed_column_log_sizes: {cols(p['LEAF_COLS'])}, root: {root(p['LEAF_ROOT'])} }},
    level1: PinnedNodeLayer {{ trace_log_size: {p['LEVEL1_TRACE']}, preprocessed_column_log_sizes: {cols(p['LEVEL1_COLS'])}, roots: {roots7(p['level1_roots'])} }},
    fold: PinnedNodeLayer {{ trace_log_size: {p['FOLD_TRACE']}, preprocessed_column_log_sizes: {cols(p['FOLD_COLS'])}, roots: {roots7(p['fold_roots'])} }},
    node_target: PinnedComponentSizes {{ eq: {nt[0]}, qm31_ops: {nt[1]}, m31_to_u32: {nt[2]}, triple_xor: {nt[3]}, blake_g_gate: {nt[4]} }},
    unpacker: PinnedUnpacker {{
        pcs: PcsConfig {{ pow_bits: {pcs[0]}, fri_config: FriConfig {{ log_blowup_factor: {pcs[1]}, log_last_layer_degree_bound: {pcs[2]}, n_queries: {pcs[3]}, fold_step: {pcs[4]} }}, lifting_log_size: Some({pcs[5]}) }},
        n_outputs: {p['UNPACKER_NOUT']},
        preprocessed_column_log_sizes: {cols(p['UNPACKER_COLS'])},
        root: {root(p['UNPACKER_ROOT'])},
    }},
    recursion_log_blowup: RECURSION_LOG_BLOWUP,
    leaf_log_blowup: RECURSION_LOG_BLOWUP,
    fold_arity: FOLD_ARITY,
    n_leaves: {n_leaves},
}};"""


def main():
    global n_leaves
    log = sys.argv[1]
    rs = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), "..", "src", "recursion_consts.rs")
    name, pt = parse(log)
    if pt is None:
        sys.exit("capture log has no @@POINT")
    k = int(name.split("N")[0][1:])   # K<k>N<n>
    n_leaves = int(name.split("N")[-1])
    # Env RECURSION_SHARD_SHOTS (>0) overrides — matches the capture's `capture_shots`, so a
    # smaller-memory GPU (2^25 shard) pins consistent consts; else the 2^26 default formula.
    shots = int(os.environ.get("RECURSION_SHARD_SHOTS") or 0) or (1 << 26) // (k * N_GATES)
    config = gen_config(pt)
    body = (
        f"pub const CAPTURED_K: usize = {k};\n"
        f"pub const CAPTURED_SHOTS: usize = {shots};\n"
        f"pub const CAPTURED_N: usize = {n_leaves};\n\n"
        f"{config}"
    )
    src = open(rs).read()
    begin_marker, end_marker = "// <<GENERATED CONSTS BEGIN", "// <<GENERATED CONSTS END>>"
    b, e = src.index(begin_marker), src.index(end_marker)
    # Keep the whole BEGIN comment header (all leading `//` lines, up to the first blank line).
    b_eol = src.index("\n\n", b) + 1
    new = src[:b_eol] + "\n" + body + "\n" + src[e:]
    open(rs, "w").write(new)
    print(f"regenerated RECURSION_CONFIG for k={k} (shots={shots}, N={n_leaves}) in {os.path.abspath(rs)}")


main()
