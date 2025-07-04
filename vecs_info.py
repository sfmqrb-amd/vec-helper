#!/usr/bin/env python3
"""
vecs_info.py

A CLI tool to read .fvecs, .bvecs, .bin, .u8bin or .i8bin files and report:
  - Number of vectors
  - Dimensionality

Use `--stat` to compute:
  - Min/Max L2 norm
  - Mean/Std of norms
  - Aspect ratio (max_norm / min_norm)
  - Global Min/Max element value

File formats:
  - .fvecs/.bvecs: per-vector 4-byte header (int32 dim) + data
  - .bin/.fbin: global header (uint32 N, uint32 D) + float32 data
  - .u8bin: global header + uint8 data
  - .i8bin: global header + int8 data

Usage:
  python3 vecs_info.py <file> [--stat]
"""
import sys, os, struct, math, argparse

def parse_args():
    p = argparse.ArgumentParser(description="Read vector files and report info")
    p.add_argument('file', help='Path to .fvecs/.bvecs/.bin/.u8bin/.i8bin file')
    p.add_argument('--stat', action='store_true', help='Compute additional statistics')
    p.add_argument('--idx', type=int, default=-1, help='Print vector at index (0-based)')
    return p.parse_args()


def compute_stats(f, count, dim, fmt, size, per_vec_header):
    min_norm = float('inf'); max_norm = 0.0
    sum_norm = 0.0; sum_sq_norm = 0.0
    vals_min = float('inf'); vals_max = float('-inf')
    actual = 0
    for i in range(count):
        if per_vec_header:
            hdr = f.read(4)
            if not hdr: break
            d = struct.unpack('i', hdr)[0]
            if d != dim:
                print(f"Warning: dim mismatch at vector {i}: expected {dim}, got {d}")
        data = f.read(size * dim)
        if len(data) < size * dim: break
        arr = struct.unpack(f'{dim}{fmt}', data)
        # value range
        for v in arr:
            if v < vals_min: vals_min = v
            if v > vals_max: vals_max = v
        # L2 norm
        sq = sum((float(v) ** 2) for v in arr)
        norm = math.sqrt(sq)
        sum_norm += norm; sum_sq_norm += norm * norm
        if norm < min_norm: min_norm = norm
        if norm > max_norm: max_norm = norm
        actual += 1
    if actual == 0:
        print('No vectors for statistics.')
        return
    mean = sum_norm / actual
    var = sum_sq_norm / actual - mean*mean
    std = math.sqrt(var) if var>0 else 0.0
    aspect = max_norm/min_norm if min_norm>0 else float('inf')
    print(f"Statistics over {actual} vectors:")
    print(f"  Min norm:    {min_norm:.6f}")
    print(f"  Max norm:    {max_norm:.6f}")
    print(f"  Mean norm:   {mean:.6f}")
    print(f"  Std norm:    {std:.6f}")
    print(f"  Aspect ratio (max/min): {aspect:.6f}")
    print(f"  Min value:   {vals_min:.6f}")
    print(f"  Max value:   {vals_max:.6f}")

def print_vector(f, idx, count, dim, fmt, size, per_vec_header, header_size):
    if idx < 0 or idx >= count:
        sys.exit(f"Error: index {idx} out of range [0, {count})")
    # compute offset
    if per_vec_header:
        entry = header_size + dim * size
        offset = idx * entry + header_size
    else:
        offset = header_size + idx * dim * size
    f.seek(offset)
    data = f.read(dim * size)
    if len(data) < dim * size:
        sys.exit(f"Error: could not read vector at index {idx}")
    arr = struct.unpack(f'{dim}{fmt}', data)
    print(f"Vector[{idx}]:")
    print(arr)


def main():
    args = parse_args(); fname = args.file
    if not os.path.isfile(fname): sys.exit(f"Error: not found: {fname}")
    lower = fname.lower()
    # determine format
    if lower.endswith('.fvecs'):
        global_hdr = False; per_vec_hdr = True; fmt='f'; sz=4
    elif lower.endswith('.bvecs'):
        global_hdr = False; per_vec_hdr = True; fmt='B'; sz=1
    elif lower.endswith('.bbin'):
        global_hdr = True; per_vec_hdr = False; fmt='B'; sz=1
    elif lower.endswith('.ibin'):
        global_hdr = True; per_vec_hdr = False; fmt='b'; sz=1
    elif lower.endswith('.fbin') or lower.endswith('.fbin'):
        global_hdr = True; per_vec_hdr = False; fmt='f'; sz=4
    else:
        sys.exit('Error: unsupported extension')
    size = os.path.getsize(fname)
    with open(fname,'rb') as f:
        if global_hdr:
            hdr = f.read(8)
            if len(hdr)<8: sys.exit('Error: incomplete header')
            num, dim = struct.unpack('II', hdr)
            header = 8
        else:
            h = f.read(4)
            if len(h)<4: sys.exit('Error: incomplete header')
            dim = struct.unpack('i', h)[0]
            num = size // (4 + dim*sz)
            header = 4
    print(f"Number of vectors: {num}")
    print(f"Dimensionality: {dim}")
    if args.stat:
        with open(fname,'rb') as f:
            f.seek(header)
            compute_stats(f, num, dim, fmt, sz, per_vec_hdr)

    if args.idx >= 0:
        with open(fname,'rb') as f:
            print_vector(f, args.idx, num, dim, fmt, sz, per_vec_hdr, header)

if __name__=='__main__':
    main()

