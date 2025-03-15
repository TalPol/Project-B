import os
import numpy as np
import sys

OUTDIR = 'squigulator/dataset/train/npy/ATCC_BAA-679__202309/dataset'
DIRS = [
    'squigulator/dataset/train/npy/ATCC_BAA-679__202309/batch_1',
    'squigulator/dataset/train/npy/ATCC_BAA-679__202309/batch_2',
    'squigulator/dataset/train/npy/ATCC_BAA-679__202309/batch_3',
    'squigulator/dataset/train/npy/ATCC_BAA-679__202309/batch_4',
    'squigulator/dataset/train/npy/ATCC_BAA-679__202309/batch_5',
    'squigulator/dataset/train/npy/ATCC_BAA-679__202309/batch_6',
    'squigulator/dataset/train/npy/ATCC_BAA-679__202309/batch_7',
    'squigulator/dataset/train/npy/ATCC_BAA-679__202309/batch_8',
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
    LENS.append(np.load(os.path.join(directory, 'reference_lengths.npy')))
    REFS.append(np.load(os.path.join(directory, 'references.npy')))
    CHUNKS.append(np.load(os.path.join(directory, 'chunks.npy')))

lens = np.concatenate(LENS)
refs, _ = rectify([x for ref in REFS for x in ref])
chunks = np.concatenate(CHUNKS, axis=0)

np.save(os.path.join(OUTDIR, 'reference_lengths.npy'), lens)
np.save(os.path.join(OUTDIR, 'references.npy'), refs)
np.save(os.path.join(OUTDIR, 'chunks.npy'), chunks)
