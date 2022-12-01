#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danylo Kovalenko
import os.path
from math import ceil
from pathlib import Path
from typing import List
from PIL import Image

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def read_and_split_images(num_workers: int, src_dir: Path) -> List[List[Path | None]]:
    data_raw = [src_dir / str(p)
                for p in os.listdir(src_dir)
                if str(p).endswith(".png")]
    pad_size = len(data_raw) - (len(data_raw) // num_workers)
    slice_size = ceil(len(data_raw) / num_workers)
    data_raw += [None for _ in range(pad_size)]
    cur_idx = 0

    res = []
    for i in range(num_workers):
        res.append(data_raw[cur_idx: cur_idx + slice_size])
        cur_idx += slice_size
    return res


def process_img(src, dst, target_dims=(500, 500)):
    img = Image.open(src)
    # Resize.LANCZOS == 1
    img = img.resize(target_dims, 1)
    img.save(dst)


def worker(data: List[Path | None], dst_dir: Path):
    for src_img_path in data:
        if src_img_path is None:
            continue
        
        dst = dst_dir / src_img_path.name
        process_img(src_img_path, dst)


def main(src_dir: Path, dst_dir: Path):
    workers = comm.Get_size()
    data = None
    if rank == 0:
        data = read_and_split_images(num_workers=workers, src_dir=src_dir)
    data = comm.scatter(data, root=0)
    worker(data, dst_dir)


if __name__ == '__main__':
    import sys
    if not len(sys.argv) == 3:
        exit("Usage: python3 resizer.py [path/to/images/dir] [path/to/dst/dir]")
    src_dir, dst_dir = sys.argv[1:3]
    main(Path(src_dir).absolute(), Path(dst_dir).absolute())