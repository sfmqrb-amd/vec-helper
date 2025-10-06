#!/usr/bin/env python3
"""
make_gt_multithreaded.py

Compute exact ground-truth k-NN (L2 or cosine) for a base and query set in a batched,
multithreaded fashion to reduce peak memory usage.

Supports formats: .fvecs, .bvecs, .ivecs, .bin/.fbin, .bbin, .u8bin, .ibin

Output: <out>.bin with header [uint32 Q][uint32 K], then Q*K uint32 indices,
then Q*K float32 distances or similarities.

Usage:
  python3 make_gt_multithreaded.py \
    --base BASE_FILE --query QUERY_FILE --out OUT_FILE --K K \
    [--batch_size BATCH] [--threads THREADS] [--metric {l2,cosine}]
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

def process_batch_l2_fast(batch_idx, q_batch, base, base_sq, K):
    """
    Compute exact K-NN (Euclidean) for q_batch against base.
    Assumes `base_sq = np.einsum('ij,ij->i', base, base)` is precomputed once.
    Returns (batch_idx, indices, distances).
    """
    # q_batch norms
    query_sq = np.einsum('ij,ij->i', q_batch, q_batch)           # (qb,)

    # full dot‐product matrix (N × qb)
    prod = base.dot(q_batch.T)                                   # (N, qb)

    # squared distances via broadcasted identity
    d2 = base_sq[:, None] + query_sq[None, :] - 2.0 * prod       # (N, qb)
    np.clip(d2, 0.0, None, out=d2)

    # get the indices of the K smallest squared‐distances per query
    idx_part = np.argpartition(d2, K, axis=0)[:K, :]             # (K, qb)

    # gather those K×qb squared‐distances
    qb = q_batch.shape[0]
    cols = np.arange(qb)
    d2_part = d2[idx_part, cols]                                 # (K, qb)

    # sort each column of the K candidates
    order = np.argsort(d2_part, axis=0)                          # (K, qb)

    # now assemble final (qb, K) arrays
    sorted_idx   = idx_part[order, cols].T                       # (qb, K)
    sorted_dist2 = np.take_along_axis(d2_part, order, axis=0).T  # (qb, K)

    return batch_idx, sorted_idx.astype(np.uint32), np.sqrt(sorted_dist2).astype(np.float32)


def process_batch(batch_idx, q_batch, base, K, metric):
    """
    Compute k-NN or k-max-sim for a slice of queries against the full base.
    metric: 'l2' for Euclidean, 'cosine' for cosine similarity.
    Returns (batch_idx, indices, distances_or_similarities).
    """
    print(f"Processing batch {batch_idx} with {q_batch.shape[0]} queries against base of size {base.shape[0]}")
    if metric == 'l2':
        batch_idx, indices, distances = process_batch_l2_fast(batch_idx, q_batch, base, np.einsum('ij,ij->i', base, base), K)
    elif metric == 'cosine':  # cosine
        base_norm = np.linalg.norm(base, axis=1)
        query_norm = np.linalg.norm(q_batch, axis=1)
        prod = base.dot(q_batch.T)
        cos_sim = prod / (base_norm[:, None] * query_norm[None, :])
        qb = q_batch.shape[0]
        idx_part = np.argpartition(-cos_sim, K, axis=0)[:K, :]
        indices = np.empty((qb, K), dtype=np.uint32)
        distances = np.empty((qb, K), dtype=np.float32)
        for j in range(qb):
            part = idx_part[:, j]
            sims = cos_sim[part, j]
            order = np.argsort(-sims)
            sel = part[order]
            indices[j, :] = sel
            distances[j, :] = sims[order]
    elif metric == 'ip':  # inner product
        prod = base.dot(q_batch.T)
        cos_sim = prod 
        qb = q_batch.shape[0]
        idx_part = np.argpartition(-cos_sim, K, axis=0)[:K, :]
        indices = np.empty((qb, K), dtype=np.uint32)
        distances = np.empty((qb, K), dtype=np.float32)
        for j in range(qb):
            part = idx_part[:, j]
            sims = cos_sim[part, j]
            order = np.argsort(-sims)
            sel = part[order]
            indices[j, :] = sel
            distances[j, :] = sims[order]

    return batch_idx, indices, distances

def main():
    p = argparse.ArgumentParser(description="Compute exact ground-truth k-NN (batched, multithreaded)")
    p.add_argument('--base',      required=True, help='Base vectors file')
    p.add_argument('--query',     required=True, help='Query vectors file')
    p.add_argument('--out',       required=True, help='Output .bin file')
    p.add_argument('--K',         type=int, required=True, help='Number of neighbors')
    p.add_argument('--batch_size',type=int, default=1024, help='Number of queries per batch')
    p.add_argument('--threads',   type=int, default=os.cpu_count(), help='Number of worker threads')
    p.add_argument('--metric',    choices=['l2','cosine','ip'], default='l2',
                   help='Distance metric: l2 (Euclidean) or cosine similarity')
    args = p.parse_args()

    base = read_vectors(args.base)
    query = read_vectors(args.query)
    if base.shape[1] != query.shape[1]:
        sys.exit(f"Dimension mismatch: base D={base.shape[1]} vs query D={query.shape[1]}")

    Q, D = query.shape
    batches = [(i, min(i + args.batch_size, Q)) for i in range(0, Q, args.batch_size)]
    results = {}

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_batch, idx, query[start:end], base, args.K, args.metric)
                   for idx, (start, end) in enumerate(batches)]
        for fut in as_completed(futures):
            batch_idx, inds, dists = fut.result()
            results[batch_idx] = (inds, dists)

    with open(args.out, 'wb') as f:
        f.write(struct.pack('<I', Q))
        f.write(struct.pack('<I', args.K))
        for idx in range(len(batches)):
            inds, _ = results[idx]
            inds.astype('<u4').tofile(f)
        for idx in range(len(batches)):
            _, dists = results[idx]
            dists.astype('<f4').tofile(f)

    print(f"Wrote ground truth {args.K}-NN ({args.metric}) for {Q} queries (batch_size={args.batch_size}, threads={args.threads}) to {args.out}")

if __name__ == '__main__':
    main()

