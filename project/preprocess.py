import os
import numpy as np
import sys
import pod5
from pod5 import Reader
from bonito.reader import normalisation
# from Bio import SeqIO

def preprocess(fasta_file, pod5_file, output_directory):
    # Create a dictionary for translating nucleotide characters to integers
    translation_dict = {65: '1', 67: '2', 71: '3', 84: '4'}
    targets = []
    lengths = []
    chunks = []
    with open(fasta_file, 'r') as file, Reader(pod5_file) as reader:
        fasta_lines = (line.strip() for line in file if not line.startswith('>'))
        for line, read in zip(fasta_lines, reader.reads()):
            # line = line.strip()
            # Skip header lines (those starting with '>')
            #if line.startswith('>'):
             #   continue
            # Translate the sequence to integers
            target = [int(x) for x in line.translate(translation_dict)]
            # read = next(read)
            #if read.sample_count > 5000:
            #    continue

            sig = read.signal
            
            #shift, scale = normalisation(sig)
            #sig = (sig - shift) / scale
            
            targets.append(target)
            lengths.append(len(target))
            chunks.append(sig)
    '''
    with Reader(pod5_file) as reader:
        for read in reader.reads():
            sig = read.signal
            shift, scale = normalisation(sig)
            sig = (sig - shift) / scale
            chunks.append(sig)
    '''
    
    max_length = max(len(chunk) for chunk in chunks)
    chunks = np.array(
        [np.pad(chunk, (0, max_length - len(chunk)), 'constant') for chunk in chunks], dtype=np.float16
    )


    # chunks = np.array(chunks, dtype=np.float16)
    targets_ = np.zeros((chunks.shape[0], max(lengths)), dtype=np.uint8)
    for idx, target in enumerate(targets): targets_[idx, :len(target)] = target
    lengths = np.array(lengths, dtype=np.uint16)

    np.save(os.path.join(output_directory, "chunks.npy"), chunks)
    np.save(os.path.join(output_directory, "references.npy"), targets_)
    np.save(os.path.join(output_directory, "reference_lengths.npy"), lengths)
    
    sys.stderr.write(f"> written ctc training data to {output_directory}\n")
    sys.stderr.write("  - chunks.npy with shape (%s)\n" % ','.join(map(str, chunks.shape)))
    sys.stderr.write("  - references.npy with shape (%s)\n" % ','.join(map(str, targets_.shape)))
    sys.stderr.write("  - reference_lengths.npy shape (%s)\n" % ','.join(map(str, lengths.shape)))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write("Usage: python preprocess.py <fasta_file> <pod5_file> <output_directory>\n")
        sys.exit(1)

    fasta_file = sys.argv[1]
    pod5_file = sys.argv[2]
    output_directory = sys.argv[3]

    preprocess(fasta_file, pod5_file, output_directory)




'''
OUTDIR = '/tmp'
DIRS = [
    '/data/p1',
    '/data/p2',
    '/data/p3',
    '/data/p4',
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
'''

