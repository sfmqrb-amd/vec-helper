#!/usr/bin/env python3
"""
make_gt.py

Compute exact ground-truth k-NN (L2) for a base and query set.
Supports formats: .fvecs, .bvecs, .ivecs, .bin/.fbin, .bbin, .u8bin, .ibin

Output: <out>.bin with header [uint32 Q][uint32 K], then Q*K uint32 indices, then Q*K float32 distances.

Usage:
  python3 make_gt.py --base BASE_FILE --query QUERY_FILE --out OUT_FILE --K K
"""
import argparse, os, struct, sys
import numpy as np

def read_vectors(path):
    """
    Load vectors from various file types into a (N, D) float32 numpy array.
    """
    lower = path.lower()
    if lower.endswith('.fvecs') or lower.endswith('.bvecs') or lower.endswith('.ivecs'):
        with open(path, 'rb') as f:
            raw = f.read()
        D = struct.unpack_from('<i', raw, 0)[0]
        if lower.endswith('.fvecs'):
            elem_size, fmt = 4, '<f'
        elif lower.endswith('.bvecs'):
            elem_size, fmt = 1, '<B'
        else:
            elem_size, fmt = 4, '<i'
        rec_size = 4 + D * elem_size
        N = len(raw) // rec_size
        dtype = np.dtype([('dim', '<i4'), ('vec', fmt, D)])
        arr = np.frombuffer(raw, dtype=dtype, count=N)
        return arr['vec'].astype(np.float32, copy=False)
    elif lower.endswith(('.bin', '.fbin', '.bbin', '.u8bin', '.ibin')):
        header = np.fromfile(path, dtype='<u4', count=2)
        N, D = int(header[0]), int(header[1])
        if lower.endswith(('.bin', '.fbin')):
            dt, elem_size = '<f4', 4
        elif lower.endswith(('.bbin', '.u8bin')):
            dt, elem_size = 'u1', 1
        else:
            dt, elem_size = 'i1', 1
        data = np.memmap(path, mode='r', dtype=dt, offset=8, shape=(N, D))
        return data.astype(np.float32, copy=False)
    else:
        sys.exit(f"Unsupported extension: {path}")


def compute_ground_truth(base, query, K):
    """
    Compute exact k-NN (L2) between base (N,D) and query (Q,D).
    Returns (indices, distances) each shape (Q, K).
    """
    base_sq = np.sum(base * base, axis=1)  # (N,)
    query_sq = np.sum(query * query, axis=1)  # (Q,)
    prod = base.dot(query.T)  # (N, Q)
    # squared distances, clamp to >=0 to avoid negative due to FP error
    d2 = base_sq[:, None] + query_sq[None, :] - 2.0 * prod
    np.clip(d2, 0.0, None, out=d2)

    Q, N = query.shape[0], base.shape[0]
    # find K smallest per column
    idx_part = np.argpartition(d2, K, axis=0)[:K, :]  # (K, Q)
    indices = np.empty((Q, K), dtype=np.uint32)
    distances = np.empty((Q, K), dtype=np.float32)
    for j in range(Q):
        part = idx_part[:, j]
        dvals = d2[part, j]
        order = np.argsort(dvals)
        sel = part[order]
        indices[j, :] = sel
        # sqrt of non-negative values yields valid distances
        distances[j, :] = np.sqrt(dvals[order])
    print(distances)
    return indices, distances


def write_ground_truth(path, indices, distances):
    """
    Write ground truth: [Q][K] header, then Q*K indices, then Q*K distances.
    """
    Q, K = indices.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', Q))
        f.write(struct.pack('<I', K))
        indices.astype('<u4').tofile(f)
        distances.astype('<f4').tofile(f)


def main():
    p = argparse.ArgumentParser(description="Compute exact ground-truth k-NN")
    p.add_argument('--base',  required=True, help='Base vectors file')
    p.add_argument('--query', required=True, help='Query vectors file')
    p.add_argument('--out',   required=True, help='Output .bin file')
    p.add_argument('--K',     type=int, required=True, help='Number of neighbors')
    args = p.parse_args()

    base = read_vectors(args.base)
    query = read_vectors(args.query)
    if base.shape[1] != query.shape[1]:
        sys.exit(f"Dimension mismatch: base D={base.shape[1]} vs query D={query.shape[1]}")

    indices, distances = compute_ground_truth(base, query, args.K)
    write_ground_truth(args.out, indices, distances)
    print(f"Wrote ground truth {args.K}-NN for {query.shape[0]} queries to {args.out}")

if __name__ == '__main__':
    main()

