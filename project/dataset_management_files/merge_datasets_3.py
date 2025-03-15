import os
import numpy as np
import mmap
import gc

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

def rectify(ragged_array, max_len=None):
    lengths = np.array([len(x) for x in ragged_array], dtype=np.uint16)
    padded = np.zeros((len(ragged_array), max_len or np.max(lengths)), dtype=ragged_array[0].dtype)
    for i, x in enumerate(ragged_array):
        padded[i, :len(x)] = x
    return padded, lengths

# Process reference lengths in append mode
all_refs_len = []
for directory in DIRS:
    ref_lens = np.load(os.path.join(directory, 'references_length.npy'), allow_pickle=False)
    all_refs_len.extend(ref_lens)
lens, _ = rectify(all_refs_len)
np.save(os.path.join(OUTDIR, 'references_length.npy'), lens)
del all_refs_len, lens
gc.collect()

# Process references
all_refs = []
for directory in DIRS:
    refs = np.load(os.path.join(directory, 'references.npy'), allow_pickle=False)
    all_refs.extend(refs)
refs, _ = rectify(all_refs)
np.save(os.path.join(OUTDIR, 'references.npy'), refs)
del all_refs, refs
gc.collect()

# Process chunks using memory mapping
chunk_shapes = [np.load(os.path.join(d, 'chunks.npy'), mmap_mode='r').shape for d in DIRS]
total_chunks = sum(s[0] for s in chunk_shapes)
chunk_shape = chunk_shapes[0][1:]  # Assumes all chunk shapes are the same

# Create memory-mapped array for writing
chunks_out = np.memmap(os.path.join(OUTDIR, 'chunks.npy'), dtype=np.float32, mode='w+', shape=(total_chunks, *chunk_shape))

idx = 0
for directory in DIRS:
    chunks = np.load(os.path.join(directory, 'chunks.npy'), mmap_mode='r')
    chunks_out[idx:idx+len(chunks)] = chunks
    idx += len(chunks)
    
    # Drop cache for this file
    with open(os.path.join(directory, 'chunks.npy'), 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mm.madvise(mmap.MADV_DONTNEED)  # Tell kernel to drop this from page cache
        mm.close()

del chunks_out
gc.collect()
