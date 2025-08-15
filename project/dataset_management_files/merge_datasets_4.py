import os
import numpy as np
import sys
import gc

OUTDIR = 'squigulator/dataset/train/npy/dataset'
DIRS = [
    'squigulator/dataset/train/npy/ATCC_BAA-679__202309_1',
    'squigulator/dataset/train/npy/ATCC_BAA-679__202309_2',
    'squigulator/dataset/train/npy/ATCC_BAA-679__202309_3',
    'squigulator/dataset/train/npy/ATCC_BAA-679__202309_4',
    'squigulator/dataset/train/npy/ATCC_BAA-679__202309_5',
    'squigulator/dataset/train/npy/ATCC_BAA-679__202309_6',
    'squigulator/dataset/train/npy/ATCC_BAA-679__202309_7',
    'squigulator/dataset/train/npy/ATCC_BAA-679__202309_8',
]

LENS = []
REFS = []
CHUNKS = []

def rectify(ragged_array, max_len=None):
    lengths = np.array([len(x) for x in ragged_array], dtype=np.uint16)
    padded = np.zeros((len(ragged_array), max_len or np.max(lengths)), dtype=ragged_array[0].dtype)
    for x, y in zip(ragged_array, padded):
        y[:len(x)] = x
    return padded, lengths


for directory in DIRS:
    LENS.append(np.load(os.path.join(directory, 'reference_lengths.npy'), mmap_mode='r'))
    

lens = np.concatenate(LENS)
np.save(os.path.join(OUTDIR, 'reference_lengths.npy'), lens)
del lens, LENS
for directory in DIRS:
    REFS.append(np.load(os.path.join(directory, 'references.npy'), mmap_mode='r'))

refs, _ = rectify([x for ref in REFS for x in ref])
np.save(os.path.join(OUTDIR, 'references.npy'), refs)
del refs, REFS

for directory in DIRS:
    CHUNKS.append(np.load(os.path.join(directory, 'chunks.npy'), mmap_mode='r'))

chunks = np.concatenate(CHUNKS, axis=0)
np.save(os.path.join(OUTDIR, 'chunks.npy'), chunks)
del chunks, CHUNKS
gc.collect()