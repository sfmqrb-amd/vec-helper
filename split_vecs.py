#!/usr/bin/env python3
"""
split_vecs.py

CLI tool to randomly sample N vectors from datasets in formats:
  .fvecs, .bvecs, .ivecs, .fbin/.bin, .ibin, .bbin

Usage:
  python3 split_vecs.py <input_file> <output_file> <N> [--seed SEED] [--both --rest REST_FILE]
"""
import sys, os, struct, random, argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly sample N vectors and optionally split the rest"
    )
    parser.add_argument('input', help='Path to input vector file')
    parser.add_argument('output', help='Path for output sampled vectors')
    parser.add_argument('N', type=int, help='Number of vectors to sample')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--both', action='store_true', help='Also output the rest of the vectors')
    parser.add_argument('--rest', help='Path for the rest dataset (required with --both)')
    return parser.parse_args()


def determine_format(path):
    lower = path.lower()
    if lower.endswith('.fvecs'):
        return dict(global_hdr=False, per_vec_hdr=True, fmt='f', size=4, header_size=4)
    if lower.endswith('.bvecs'):
        return dict(global_hdr=False, per_vec_hdr=True, fmt='B', size=1, header_size=4)
    if lower.endswith('.ivecs'):
        return dict(global_hdr=False, per_vec_hdr=True, fmt='i', size=4, header_size=4)
    if lower.endswith('.fbin') or lower.endswith('.bin'):
        return dict(global_hdr=True, per_vec_hdr=False, fmt='f', size=4, header_size=8)
    if lower.endswith('.ibin'):
        return dict(global_hdr=True, per_vec_hdr=False, fmt='b', size=1, header_size=8)
    if lower.endswith('.bbin'):
        return dict(global_hdr=True, per_vec_hdr=False, fmt='B', size=1, header_size=8)
    sys.exit('Error: unsupported file extension')


def read_header(f, cfg):
    if cfg['global_hdr']:
        hdr = f.read(cfg['header_size'])
        if len(hdr) < cfg['header_size']:
            sys.exit('Error: incomplete file header')
        num, dim = struct.unpack('II', hdr)
    elif cfg['per_vec_hdr']:
        hdr = f.read(cfg['header_size'])
        if len(hdr) < cfg['header_size']:
            sys.exit('Error: incomplete per-vector header')
        dim = struct.unpack('i', hdr)[0]
        entry = cfg['header_size'] + dim * cfg['size']
        f.seek(0, os.SEEK_END)
        total = f.tell() // entry
        num = total
        f.seek(0)
    else:
        sys.exit('Error: cannot determine header type')
    return num, dim


def sample_indices(total, N, seed=None):
    if N > total:
        sys.exit(f'Error: sample size N={N} larger than total vectors {total}')
    if seed is not None:
        random.seed(seed)
    return sorted(random.sample(range(total), N))


def write_samples(in_path, out_path, cfg, indices, dim):
    with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
        if cfg['global_hdr']:
            fout.write(struct.pack('II', len(indices), dim))
            data_offset = cfg['header_size']
            rec_size = dim * cfg['size']
            for idx in indices:
                fin.seek(data_offset + idx * rec_size)
                fout.write(fin.read(rec_size))
        else:
            rec_size = cfg['header_size'] + dim * cfg['size']
            for idx in indices:
                fin.seek(idx * rec_size)
                fout.write(fin.read(rec_size))


def main():
    args = parse_args()
    if args.both and not args.rest:
        sys.exit('Error: --rest must be provided when using --both')

    cfg = determine_format(args.input)
    with open(args.input, 'rb') as f:
        total, dim = read_header(f, cfg)

    # sample and rest indices
    sampled = sample_indices(total, args.N, args.seed)
    write_samples(args.input, args.output, cfg, sampled, dim)
    print(f'Sampled {len(sampled)} vectors into {args.output}')

    if args.both:
        rest_set = set(range(total)) - set(sampled)
        rest = sorted(rest_set)
        write_samples(args.input, args.rest, cfg, rest, dim)
        print(f'Wrote remaining {len(rest)} vectors into {args.rest}')


if __name__ == '__main__':
    main()

