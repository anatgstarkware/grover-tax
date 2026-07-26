
#define M31_P 0x7FFFFFFFu

// ---- M31 (p = 2^31 - 1) ----
__device__ __forceinline__ unsigned m31_add(unsigned a, unsigned b) {
    unsigned r = a + b;
    unsigned reduced = (r & M31_P) + (r >> 31);
    return reduced == M31_P ? 0u : reduced;
}
__device__ __forceinline__ unsigned m31_sub(unsigned a, unsigned b) {
    unsigned r = a - b;
    return r + (M31_P & -(r >> 31));
}
__device__ __forceinline__ unsigned m31_neg(unsigned a) {
    return (M31_P - a) * (a != 0u);
}
__device__ __forceinline__ unsigned m31_mul(unsigned a, unsigned b) {
    unsigned long long prod = (unsigned long long)a * (unsigned long long)b;
    unsigned lo = (unsigned)(prod & M31_P);
    unsigned hi = (unsigned)(prod >> 31);
    return m31_add(lo, hi);
}
__device__ __forceinline__ unsigned m31_sqr(unsigned a) { return m31_mul(a, a); }
__device__ unsigned m31_pow(unsigned base, unsigned exp) {
    unsigned result = 1u, b = base;
    while (exp > 0u) { if (exp & 1u) result = m31_mul(result, b); b = m31_sqr(b); exp >>= 1; }
    return result;
}
// a^(p-2) = a^(2^31 - 3).
__device__ __forceinline__ unsigned m31_inv(unsigned a) { return m31_pow(a, 0x7FFFFFFDu); }

// ---- CM31 = M31[i]/(i^2 + 1) ----
struct cm31 { unsigned a; unsigned b; };
__device__ __forceinline__ cm31 cm31_add(cm31 x, cm31 y) { return {m31_add(x.a,y.a), m31_add(x.b,y.b)}; }
__device__ __forceinline__ cm31 cm31_sub(cm31 x, cm31 y) { return {m31_sub(x.a,y.a), m31_sub(x.b,y.b)}; }
__device__ __forceinline__ cm31 cm31_mul(cm31 x, cm31 y) {
    return { m31_sub(m31_mul(x.a,y.a), m31_mul(x.b,y.b)),
             m31_add(m31_mul(x.a,y.b), m31_mul(x.b,y.a)) };
}
__device__ __forceinline__ cm31 cm31_inv(cm31 t) {
    unsigned factor = m31_inv(m31_add(m31_mul(t.a,t.a), m31_mul(t.b,t.b)));
    return { m31_mul(t.a, factor), m31_mul(m31_neg(t.b), factor) };
}

// ---- QM31 = CM31[j]/(j^2 - (2+i)), R = (2,1) ----
struct qm31 { cm31 a; cm31 b; };
__device__ __forceinline__ qm31 qm31_add(qm31 x, qm31 y) { return {cm31_add(x.a,y.a), cm31_add(x.b,y.b)}; }
__device__ __forceinline__ qm31 qm31_sub(qm31 x, qm31 y) { return {cm31_sub(x.a,y.a), cm31_sub(x.b,y.b)}; }
__device__ __forceinline__ qm31 qm31_mul(qm31 x, qm31 y) {
    // (a + b·j)(c + d·j) = (a·c + R·b·d) + (a·d + b·c)·j
    cm31 R = {2u, 1u};
    cm31 ac = cm31_mul(x.a, y.a);
    cm31 bd = cm31_mul(x.b, y.b);
    cm31 ad = cm31_mul(x.a, y.b);
    cm31 bc = cm31_mul(x.b, y.a);
    return { cm31_add(ac, cm31_mul(R, bd)), cm31_add(ad, bc) };
}
__device__ __forceinline__ qm31 qm31_mul_m31(qm31 x, unsigned s) {
    return { { m31_mul(x.a.a,s), m31_mul(x.a.b,s) }, { m31_mul(x.b.a,s), m31_mul(x.b.b,s) } };
}
__device__ __forceinline__ qm31 qm31_inv(qm31 t) {
    // (a + b·j)^{-1} = (a − b·j) / (a^2 − (2+i)·b^2).
    cm31 b2 = cm31_mul(t.b, t.b);
    cm31 ib2 = { m31_neg(b2.b), b2.a };                       // i · b2
    cm31 denom = cm31_sub(cm31_mul(t.a, t.a), cm31_add(cm31_add(b2, b2), ib2));
    cm31 di = cm31_inv(denom);
    cm31 nb = { m31_neg(t.b.a), m31_neg(t.b.b) };
    return { cm31_mul(t.a, di), cm31_mul(nb, di) };
}
__device__ __forceinline__ unsigned qm31_zero_get() { return 0u; }

// combine(values) = ( Σ_i alpha_powers[i] · values[i] ) − z   (QM31).
// `ap` holds GATE_REL_WIDTH QM31s flattened as 4 unsigned each.
__device__ __forceinline__ qm31 logup_combine(
    const unsigned* vals, int n,
    unsigned z0, unsigned z1, unsigned z2, unsigned z3,
    const unsigned* ap)
{
    qm31 acc = { {0u,0u}, {0u,0u} };
    for (int i = 0; i < n; i++) {
        qm31 apw = { { ap[i*4+0], ap[i*4+1] }, { ap[i*4+2], ap[i*4+3] } };
        acc = qm31_add(acc, qm31_mul_m31(apw, vals[i]));
    }
    qm31 z = { {z0,z1}, {z2,z3} };
    return qm31_sub(acc, z);
}

// K4a: per-row (num, denom) for one of the 5 batches (3 qubitmem pairs + 2 rc/program pairs). Reads
// K1's 19 main-trace columns (column-major in `cols`). Outputs interleaved per row:
// num[row*4+j], denom[row*4+j].
//
// Column layout (cell_at, 19 cols, ACCESS_BLOCK=4): 0..4 opcode one-hots;
//   target addr=4,prev_ts=5,v=6,d=7;
//   ctrl_a addr=8,prev_ts=9,v=10,d=11;
//   ctrl_b addr=12,prev_ts=13,v=14,d=15; ab=16,fire=17,delta=18.
// ts (= pc+1) and the target's v_after (= v_before+delta) are INLINED here (recomputed from pc / the
// v+delta columns), NOT read as columns.
// Relation tags: QUBITMEM=1, RC=2, PROGRAM=5. Widths: qubitmem tuple=5, rc tuple=2 (TAG_RC, d),
// program tuple=6 (rel width = GATE_REL_WIDTH = 6). Batch order MUST match gen_main_interaction /
// circuit_statement EXACTLY (the 10-entry stream folded into 5 pairs by finalize_logup_in_pairs):
//   case 0..2 : 3 qubitmem pairs   (target / ctrl_a / ctrl_b : Use[+active] / Yield[-active])
//   case 3    : rc target d [+enabler]   / rc ctrl_a d [+a_active]
//   case 4    : rc ctrl_b d [+b_active]  / program SINGLETON [+enabler]
// A batch is a SINGLETON when n1 == 0 (only v0/m0 contribute: num = m0, den = d0); else a pair. (No
// singleton batches remain in this layout — the program now pairs with rc ctrl_b d in case 4.)
extern "C" __global__ void logup_col_gen(
    const unsigned* __restrict__ cols,
    unsigned long padded_rows,
    unsigned z0, unsigned z1, unsigned z2, unsigned z3,
    const unsigned* __restrict__ ap,
    unsigned pair_id,
    unsigned n_gates,
    const unsigned long* __restrict__ dims, // dims[0]=real_rows, dims[1]=shot_stride (=k*n_gates)
    unsigned* __restrict__ num,
    unsigned* __restrict__ denom)
{
    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= padded_rows) return;
    unsigned long real_rows   = dims[0];
    unsigned long shot_stride = dims[1];
    #define COL(c) cols[(unsigned long)(c) * padded_rows + row]

    unsigned v0[6]; int n0 = 0;
    unsigned v1[6]; int n1 = 0;
    unsigned m0 = 0u, m1 = 0u; int s0 = 1, s1 = 1;

    // enabler / shot_id / pc are tree0 (POSITIONAL). K4 recomputes them from `row` — identical
    // values to the preprocessed columns (enabler = real-row indicator, shot_id = row / shot_stride,
    // pc = row % shot_stride). pc feeds only the program term's pc_in_prog = pc % n_gates.
    unsigned enabler  = (row < real_rows) ? 1u : 0u;
    unsigned shot_id  = (row < real_rows) ? (unsigned)(row / shot_stride) : 0u;
    unsigned pc       = (row < real_rows) ? (unsigned)(row % shot_stride) : 0u;
    unsigned a_active = m31_add(COL(2), COL(3));   // is_cnot + is_toffoli
    unsigned b_active = COL(3);                     // is_toffoli
    // ts = pc + 1 (INLINED, shared by all accesses of the step; not a column).
    unsigned ts = m31_add(pc, 1u);
    // v_after = v_before + delta = t.v(col6) + delta(col18) (INLINED target write value; not a column).
    unsigned v_after = m31_add(COL(6), COL(18));

    switch (pair_id) {
    case 0: // qubitmem target Use (+enabler): [1, shot, t.addr, t.prev_ts, t.v_before]
            //          target Yield (-enabler): [1, shot, t.addr, ts=pc+1,  v_after]
        v0[0]=1u; v0[1]=shot_id; v0[2]=COL(4); v0[3]=COL(5); v0[4]=COL(6); n0=5;   // t.addr,t.prev_ts,t.v
        v1[0]=1u; v1[1]=shot_id; v1[2]=COL(4); v1[3]=ts;     v1[4]=v_after; n1=5;   // ts=pc+1, v_after=v+delta
        m0=enabler; s0=1; m1=enabler; s1=-1; break;
    case 1: // qubitmem ctrl_a Use (+a_active): [1, shot, a.addr, a.prev_ts, a.v]
            //          ctrl_a Yield (-a_active): [1, shot, a.addr, ts=pc+1,  a.v]  (read propagates)
        v0[0]=1u; v0[1]=shot_id; v0[2]=COL(8); v0[3]=COL(9);  v0[4]=COL(10); n0=5; // a.addr,a.prev_ts,a.v
        v1[0]=1u; v1[1]=shot_id; v1[2]=COL(8); v1[3]=ts;      v1[4]=COL(10); n1=5;
        m0=a_active; s0=1; m1=a_active; s1=-1; break;
    case 2: // qubitmem ctrl_b Use (+b_active): [1, shot, b.addr, b.prev_ts, b.v]
            //          ctrl_b Yield (-b_active): [1, shot, b.addr, ts=pc+1,  b.v]
        v0[0]=1u; v0[1]=shot_id; v0[2]=COL(12); v0[3]=COL(13); v0[4]=COL(14); n0=5; // b.addr,b.prev_ts,b.v
        v1[0]=1u; v1[1]=shot_id; v1[2]=COL(12); v1[3]=ts;      v1[4]=COL(14); n1=5;
        m0=b_active; s0=1; m1=b_active; s1=-1; break;
    case 3: // rc target d (+enabler): [2, t.d] / rc ctrl_a d (+a_active): [2, a.d]
        v0[0]=2u; v0[1]=COL(7);  n0=2;                                             // t.d=col7
        v1[0]=2u; v1[1]=COL(11); n1=2;                                             // a.d=col11
        m0=enabler; s0=1; m1=a_active; s1=1; break;
    case 4: // rc ctrl_b d (+b_active): [2, b.d] / program (+enabler): [5, pc%ng, opcode_scalar, t/a/b addr]
        v0[0]=2u; v0[1]=COL(15); n0=2;                                             // b.d=col15
        v1[0]=5u; v1[1]=(unsigned)((unsigned long)pc % (unsigned long)n_gates);
        v1[2]=m31_add(m31_add(COL(1), m31_mul(2u,COL(2))), m31_mul(3u,COL(3))); // opcode_scalar
        v1[3]=COL(4); v1[4]=COL(8); v1[5]=COL(12); n1=6;                        // t/a/b addr
        m0=b_active; s0=1; m1=enabler; s1=1; break;
    }

    // SINGLETON (n1 == 0): num = m0, den = d0. PAIR: num = m0*d1 + m1*d0, den = d0*d1.
    qm31 d0 = logup_combine(v0, n0, z0,z1,z2,z3, ap);
    unsigned mm0 = (s0 < 0) ? m31_neg(m0) : m0;
    qm31 qm0 = { {mm0,0u}, {0u,0u} };
    qm31 nume, den;
    if (n1 == 0) {
        nume = qm0;
        den  = d0;
    } else {
        qm31 d1 = logup_combine(v1, n1, z0,z1,z2,z3, ap);
        unsigned mm1 = (s1 < 0) ? m31_neg(m1) : m1;
        qm31 qm1 = { {mm1,0u}, {0u,0u} };
        nume = qm31_add(qm31_mul(qm0, d1), qm31_mul(qm1, d0));
        den  = qm31_mul(d0, d1);
    }

    num[row*4+0]=nume.a.a; num[row*4+1]=nume.a.b; num[row*4+2]=nume.b.a; num[row*4+3]=nume.b.b;
    denom[row*4+0]=den.a.a; denom[row*4+1]=den.a.b; denom[row*4+2]=den.b.a; denom[row*4+3]=den.b.b;
    #undef COL
}

// K4b: value = num · denom^{-1}; running sum onto previous logup column.
// `inter` holds the N_INTERACTION_COLS (20) interaction columns, column-major (logup col k coord j = (k*4+j)).
extern "C" __global__ void logup_finalize_col(
    unsigned col_k,
    unsigned long padded_rows,
    const unsigned* __restrict__ num,
    const unsigned* __restrict__ denom,
    unsigned* __restrict__ inter)
{
    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= padded_rows) return;
    qm31 nume = { { num[row*4+0], num[row*4+1] }, { num[row*4+2], num[row*4+3] } };
    qm31 den  = { { denom[row*4+0], denom[row*4+1] }, { denom[row*4+2], denom[row*4+3] } };
    qm31 value = qm31_mul(nume, qm31_inv(den));
    qm31 prev = { {0u,0u}, {0u,0u} };
    if (col_k > 0u) {
        unsigned long b = (unsigned long)(col_k - 1u) * 4u;
        prev.a.a = inter[(b+0)*padded_rows+row];
        prev.a.b = inter[(b+1)*padded_rows+row];
        prev.b.a = inter[(b+2)*padded_rows+row];
        prev.b.b = inter[(b+3)*padded_rows+row];
    }
    qm31 acc = qm31_add(value, prev);
    unsigned long b = (unsigned long)col_k * 4u;
    inter[(b+0)*padded_rows+row] = acc.a.a;
    inter[(b+1)*padded_rows+row] = acc.a.b;
    inter[(b+2)*padded_rows+row] = acc.b.a;
    inter[(b+3)*padded_rows+row] = acc.b.b;
}

__device__ __forceinline__ unsigned m31_atomic_add(unsigned* addr, unsigned val) {
    unsigned old = *addr, assumed;
    do { assumed = old; old = atomicCAS(addr, assumed, m31_add(assumed, val)); } while (assumed != old);
    return old;
}

// K4c.1: claimed_sum = Σ_row last_col (per coordinate). Block-reduce + atomic into sums[4].
extern "C" __global__ void logup_cumsum_reduce(
    unsigned long padded_rows,
    unsigned last_k,
    const unsigned* __restrict__ inter,
    unsigned* __restrict__ sums)
{
    extern __shared__ unsigned sh[];   // 4 * blockDim.x
    unsigned* s0 = &sh[0];
    unsigned* s1 = &sh[blockDim.x];
    unsigned* s2 = &sh[2*blockDim.x];
    unsigned* s3 = &sh[3*blockDim.x];
    unsigned long b = (unsigned long)last_k * 4u;
    unsigned long tid = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long stride = (unsigned long)gridDim.x * blockDim.x;
    unsigned a0=0u,a1=0u,a2=0u,a3=0u;
    for (unsigned long i = tid; i < padded_rows; i += stride) {
        a0 = m31_add(a0, inter[(b+0)*padded_rows+i]);
        a1 = m31_add(a1, inter[(b+1)*padded_rows+i]);
        a2 = m31_add(a2, inter[(b+2)*padded_rows+i]);
        a3 = m31_add(a3, inter[(b+3)*padded_rows+i]);
    }
    s0[threadIdx.x]=a0; s1[threadIdx.x]=a1; s2[threadIdx.x]=a2; s3[threadIdx.x]=a3;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0u; s >>= 1) {
        if (threadIdx.x < s) {
            s0[threadIdx.x]=m31_add(s0[threadIdx.x],s0[threadIdx.x+s]);
            s1[threadIdx.x]=m31_add(s1[threadIdx.x],s1[threadIdx.x+s]);
            s2[threadIdx.x]=m31_add(s2[threadIdx.x],s2[threadIdx.x+s]);
            s3[threadIdx.x]=m31_add(s3[threadIdx.x],s3[threadIdx.x+s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        m31_atomic_add(&sums[0], s0[0]);
        m31_atomic_add(&sums[1], s1[0]);
        m31_atomic_add(&sums[2], s2[0]);
        m31_atomic_add(&sums[3], s3[0]);
    }
}

// K4c.2: subtract cumsum_shift = claimed_sum / trace_size from the last column.
extern "C" __global__ void logup_cumsum_shift(
    unsigned long padded_rows,
    unsigned last_k,
    unsigned trace_size,
    const unsigned* __restrict__ sums,
    unsigned* __restrict__ inter)
{
    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= padded_rows) return;
    qm31 claimed = { { sums[0], sums[1] }, { sums[2], sums[3] } };
    qm31 shift = qm31_mul_m31(claimed, m31_inv(trace_size));
    unsigned long b = (unsigned long)last_k * 4u;
    inter[(b+0)*padded_rows+row] = m31_sub(inter[(b+0)*padded_rows+row], shift.a.a);
    inter[(b+1)*padded_rows+row] = m31_sub(inter[(b+1)*padded_rows+row], shift.a.b);
    inter[(b+2)*padded_rows+row] = m31_sub(inter[(b+2)*padded_rows+row], shift.b.a);
    inter[(b+3)*padded_rows+row] = m31_sub(inter[(b+3)*padded_rows+row], shift.b.b);
}

// ---- inclusive prefix sum (matches stwo inclusive_prefix_sum semantics) ----
__device__ __forceinline__ unsigned bitrev(unsigned x, unsigned bits) {
    unsigned r = 0u;
    for (unsigned i = 0u; i < bits; i++) { r = (r << 1) | (x & 1u); x >>= 1; }
    return r;
}
// In-place bit-reverse permutation of one column slice [offset, offset+n).
extern "C" __global__ void ps_bit_reverse(unsigned* col, unsigned long offset, unsigned n, unsigned bits) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned r = bitrev(idx, bits);
    if (r > idx) {
        unsigned t = col[offset+idx];
        col[offset+idx] = col[offset+r];
        col[offset+r] = t;
    }
}
// CircleDomain order -> Coset order: out[2i]=in[i], out[2i+1]=in[n-1-i].
extern "C" __global__ void ps_circle_to_coset(const unsigned* col, unsigned long offset, unsigned* tmp, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n/2u) { tmp[2u*i] = col[offset+i]; tmp[2u*i+1u] = col[offset + n - 1u - i]; }
}
// Coset order -> CircleDomain order.
extern "C" __global__ void ps_coset_to_circle(const unsigned* tmp, unsigned* col, unsigned long offset, unsigned n) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    unsigned half = n/2u;
    if (tid < half) col[offset+tid] = tmp[tid << 1];
    else { unsigned i = tid - half; col[offset+tid] = tmp[n - 1u - (i << 1)]; }
}
// Per-block inclusive scan (Hillis–Steele), in place; writes block totals to block_sums.
extern "C" __global__ void ps_block_scan(unsigned* data, unsigned* block_sums, unsigned n) {
    extern __shared__ unsigned sh[];
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned tid = threadIdx.x;
    sh[tid] = (gid < n) ? data[gid] : 0u;
    __syncthreads();
    for (unsigned off = 1u; off < blockDim.x; off <<= 1) {
        unsigned t = (tid >= off) ? sh[tid - off] : 0u;
        __syncthreads();
        sh[tid] = m31_add(sh[tid], t);
        __syncthreads();
    }
    if (gid < n) data[gid] = sh[tid];
    if (tid == blockDim.x - 1u) block_sums[blockIdx.x] = sh[tid];
}
// Add each block's exclusive offset (inclusive-scanned block sums of prior blocks).
extern "C" __global__ void ps_add_offsets(unsigned* out, const unsigned* scanned_block_sums, unsigned n) {
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n && blockIdx.x > 0u) out[gid] = m31_add(out[gid], scanned_block_sums[blockIdx.x - 1u]);
}
