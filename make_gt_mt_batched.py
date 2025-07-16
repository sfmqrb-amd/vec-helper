#!/usr/bin/env python3
"""
make_gt_multithreaded.py

Compute exact ground-truth k-NN (L2) for a base and query set in a batched,
multithreaded fashion to reduce peak memory usage.

Supports formats: .fvecs, .bvecs, .ivecs, .bin/.fbin, .bbin, .u8bin, .ibin

Output: <out>.bin with header [uint32 Q][uint32 K], then Q*K uint32 indices,
then Q*K float32 distances.

Usage:
  python3 make_gt_multithreaded.py \
    --base BASE_FILE --query QUERY_FILE --out OUT_FILE --K K \
    [--batch_size BATCH] [--threads THREADS]
"""
import argparse, os, struct, sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def read_vectors(path):
    """
    Load vectors into a (N, D) float32 numpy array or memmap.
    """
    lower = path.lower()
    if lower.endswith(('.bin', '.fbin', '.bbin', '.u8bin', '.ibin')):
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
    # fall back to small-format in-memory read
    with open(path, 'rb') as f:
        raw = f.read()
    D = struct.unpack_from('<i', raw, 0)[0]
    if lower.endswith('.fvecs'):
        elem_size, fmt = 4, '<f'
    elif lower.endswith('.bvecs'):
        elem_size, fmt = 1, '<B'
    elif lower.endswith('.ivecs'):
        elem_size, fmt = 4, '<i'
    else:
        sys.exit(f"Unsupported extension: {path}")
    rec_size = 4 + D * elem_size
    N = len(raw) // rec_size
    dtype = np.dtype([('dim', '<i4'), ('vec', fmt, D)])
    arr = np.frombuffer(raw, dtype=dtype, count=N)
    return arr['vec'].astype(np.float32, copy=False)

def process_batch(batch_idx, q_batch, base, K):
    """
    Compute k-NN for a slice of queries against the full base.
    Returns (batch_idx, indices, distances).
    """
    # precompute squares
    base_sq = np.sum(base * base, axis=1)          # (N,)
    query_sq = np.sum(q_batch * q_batch, axis=1)    # (qb,)
    prod = base.dot(q_batch.T)                      # (N, qb)
    d2 = base_sq[:, None] + query_sq[None, :] - 2.0 * prod
    np.clip(d2, 0.0, None, out=d2)

    qb = q_batch.shape[0]
    idx_part = np.argpartition(d2, K, axis=0)[:K, :]  # (K, qb)
    indices = np.empty((qb, K), dtype=np.uint32)
    distances = np.empty((qb, K), dtype=np.float32)
    for j in range(qb):
        part = idx_part[:, j]
        dvals = d2[part, j]
        order = np.argsort(dvals)
        sel = part[order]
        indices[j, :] = sel
        distances[j, :] = np.sqrt(dvals[order])
    return batch_idx, indices, distances

def main():
    p = argparse.ArgumentParser(description="Compute exact ground-truth k-NN (batched, multithreaded)")
    p.add_argument('--base',      required=True, help='Base vectors file')
    p.add_argument('--query',     required=True, help='Query vectors file')
    p.add_argument('--out',       required=True, help='Output .bin file')
    p.add_argument('--K',         type=int, required=True, help='Number of neighbors')
    p.add_argument('--batch_size',type=int, default=1024, help='Number of queries per batch')
    p.add_argument('--threads',   type=int, default=os.cpu_count(), help='Number of worker threads')
    args = p.parse_args()

    base = read_vectors(args.base)
    query = read_vectors(args.query)
    if base.shape[1] != query.shape[1]:
        sys.exit(f"Dimension mismatch: base D={base.shape[1]} vs query D={query.shape[1]}")

    Q, D = query.shape
    # prepare batch indices
    batches = [(i, min(i + args.batch_size, Q)) for i in range(0, Q, args.batch_size)]
    results = {}

    # submit jobs
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_batch, idx, query[start:end], base, args.K)
                   for idx, (start, end) in enumerate(batches)]
        for fut in as_completed(futures):
            batch_idx, inds, dists = fut.result()
            results[batch_idx] = (inds, dists)

    # write output: header, then all indices, then all distances
    with open(args.out, 'wb') as f:
        f.write(struct.pack('<I', Q))
        f.write(struct.pack('<I', args.K))
        # write indices
        for idx in range(len(batches)):
            inds, _ = results[idx]
            inds.astype('<u4').tofile(f)
        # write distances
        for idx in range(len(batches)):
            _, dists = results[idx]
            dists.astype('<f4').tofile(f)

    print(f"Wrote ground truth {args.K}-NN for {Q} queries (batch_size={args.batch_size}, threads={args.threads}) to {args.out}")

if __name__ == '__main__':
    main()

