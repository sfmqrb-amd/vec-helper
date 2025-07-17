#!/usr/bin/env python3
"""
convert_vec_format.py

Convert between .fbin (global header + raw float data) and .fvecs
(each vector prefixed by a dimension header) formats.

Usage:
    python3 convert_vec_format.py <input_file> <output_file> (--to-fvecs | --to-fbin)
"""
import sys
import struct
import os

def fbin_to_fvecs(in_path, out_path):
    # Read global header: two unsigned ints (num_vectors, dimension)
    with open(in_path, 'rb') as fin:
        header = fin.read(8)
        if len(header) != 8:
            sys.exit("Error: incomplete file header (expected 8 bytes).")
        num_vectors, dim = struct.unpack('II', header)

        with open(out_path, 'wb') as fout:
            for i in range(num_vectors):
                # Write the per-vector dimension header (int32)
                fout.write(struct.pack('i', dim))
                data = fin.read(dim * 4)
                if len(data) != dim * 4:
                    sys.exit(f"Error: vector {i} is incomplete (expected {dim*4} bytes).")
                fout.write(data)
    print(f"Converted {num_vectors} vectors (dim={dim}) from {in_path} to {out_path} in .fvecs format.")


def fvecs_to_fbin(in_path, out_path):
    vectors = []
    dim = None
    count = 0

    # Read all vectors with per-vector headers
    with open(in_path, 'rb') as fin:
        while True:
            dim_bytes = fin.read(4)
            if not dim_bytes:
                break  # EOF
            if len(dim_bytes) != 4:
                sys.exit("Error: incomplete per-vector dimension header.")
            vdim = struct.unpack('i', dim_bytes)[0]
            if dim is None:
                dim = vdim
            elif vdim != dim:
                sys.exit(f"Error: inconsistent vector dimension at index {count}: {vdim} vs {dim}.")

            data = fin.read(dim * 4)
            if len(data) != dim * 4:
                sys.exit(f"Error: vector {count} data is incomplete (expected {dim*4} bytes).")
            vectors.append(data)
            count += 1

    # Write global header and concatenated vector data
    with open(out_path, 'wb') as fout:
        fout.write(struct.pack('II', count, dim))
        for i, data in enumerate(vectors):
            fout.write(data)

    print(f"Converted {count} vectors (dim={dim}) from {in_path} to {out_path} in .fbin format.")


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 convert_vec_format.py <input> <output> (--to-fvecs | --to-fbin)")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    flag = sys.argv[3].lower()

    if flag == '--to-fvecs':
        fbin_to_fvecs(in_path, out_path)
    elif flag == '--to-fbin':
        fvecs_to_fbin(in_path, out_path)
    else:
        sys.exit("Error: unknown flag. Use --to-fvecs or --to-fbin.")

if __name__ == '__main__':
    main()

