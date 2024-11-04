import pyslow5
import pysam
import numpy as np
import os
import sys

# Constants
FIXED_LENGTH = 1000  # Target length for each trimmed signal segment
OVERLAP = 500        # Overlap in samples between consecutive segments

# Load the .slow5 file
def load_slow5_data(slow5_file_path):
    slow5_data = {}
    s5 = pyslow5.Open(slow5_file_path, 'r')
    for read in s5.seq_reads():
        read_id = read['read_id']
        raw_signal = read['signal']
        slow5_data[read_id] = raw_signal
    return slow5_data

# Parse the SAM file to extract the nucleotide mappings and coordinates
def parse_sam_file(sam_file_path):
    nucleotide_mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    sam_data = {}
    with pysam.AlignmentFile(sam_file_path, "r") as samfile:
        for read in samfile:
            read_id = read.query_name
            ss_tag = read.get_tag('SS') if read.has_tag('SS') else None
            si_tag = read.get_tag('SI') if read.has_tag('SI') else None
            if ss_tag and si_tag:
                ss_values = list(map(int, ss_tag.split(',')))
                si_values = list(map(int, si_tag.split(',')))
                nucleotides = [nucleotide_mapping.get(nuc, 0) for nuc in read.query_sequence]
                sam_data[read_id] = {
                    'nucleotides': nucleotides,
                    'ss': ss_values,
                    'si': si_values
                }
    return sam_data

# Generate overlapping segments for the signal and nucleotide sequence
def generate_overlapping_segments(signal, nucleotides, ss_values, si_values, fixed_length=FIXED_LENGTH, overlap=OVERLAP):
    segments = []
    step = fixed_length - overlap

    for i in range(0, len(ss_values) - fixed_length + 1, step):
        start = i
        end = start + fixed_length
        segment_signal = signal[ss_values[start]:ss_values[end]]
        segment_nucleotides = nucleotides[start:end]
        segment_intensity = si_values[start:end]
        
        if len(segment_signal) < fixed_length:
            continue

        segments.append((segment_signal, segment_nucleotides, segment_intensity))
    
    return segments

# Process each read, generate overlapping segments, and save as .npy
def process_reads(slow5_data, sam_data, output_dir):
    all_signals = []
    all_nucleotides = []
    all_intensities = []
    all_lengths = []

    for read_id, signal in slow5_data.items():
        if read_id in sam_data:
            nucleotides = sam_data[read_id]['nucleotides']
            ss_values = sam_data[read_id]['ss']
            si_values = sam_data[read_id]['si']
            segments = generate_overlapping_segments(signal, nucleotides, ss_values, si_values)

            for seg_signal, seg_nucleotides, seg_intensity in segments:
                all_signals.append(seg_signal)
                all_nucleotides.append(list(seg_nucleotides))
                all_intensities.append(list(seg_intensity))
                all_lengths.append(len(seg_nucleotides))

    chunks = np.array(all_signals, dtype=np.float16)
    targets_ = np.zeros((chunks.shape[0], max(all_lengths)), dtype=np.uint8)
    intensities = np.zeros((chunks.shape[0], max(all_lengths)), dtype=np.uint8)
    for idx, (target, intensity) in enumerate(zip(all_nucleotides, all_intensities)):
        targets_[idx, :len(target)] = target
        intensities[idx, :len(intensity)] = intensity
    
    lengths = np.array(all_lengths, dtype=np.uint16)
    np.save(f"{output_dir}/chunks.npy", chunks)
    np.save(f"{output_dir}/references.npy", targets_)
    np.save(f"{output_dir}/intensities.npy", intensities)
    np.save(f"{output_dir}/reference_lengths.npy", lengths)

# Main script
if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write("Usage: python preprocess.py <slow5_file> <sam_file> <output_directory>\n")
        sys.exit(1)

    slow5_file_path = sys.argv[1]
    sam_file_path = sys.argv[2]
    output_dir = sys.argv[3]

    slow5_data = load_slow5_data(slow5_file_path)
    sam_data = parse_sam_file(sam_file_path)
    
    os.makedirs(output_dir, exist_ok=True)
    process_reads(slow5_data, sam_data, output_dir)
