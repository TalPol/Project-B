import pyslow5
import pysam
import numpy as np
import os
import sys
import re
from bonito.reader import normalisation

# Constants
FIXED_LENGTH = 5000  # Target length for each trimmed signal segment
OVERLAP = 100        # Overlap in samples between consecutive segments

# Load the .slow5 file
def load_slow5_data(slow5_file_path):
    slow5_data = {}
    s5 = pyslow5.Open(slow5_file_path, 'r')
    for read in s5.seq_reads():
        read_id = read['read_id']
        raw_signal = read['signal']
        #shift, scale = normalisation(raw_signal)
        #raw_signal = (raw_signal - shift) / scale
        slow5_data[read_id] = raw_signal
    return slow5_data

# Parse the SAM file to extract the nucleotide mappings and coordinates
'''
def parse_sam_file(sam_file_path):
    nucleotide_mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    sam_data = {}

    with pysam.AlignmentFile(sam_file_path, "r") as samfile:
        for read in samfile:
            read_id = read.query_name
            # Extract ss and si tags (signal start and signal end)
            ss_tag = read.get_tag('ss') if read.has_tag('ss') else None
            si_tag = read.get_tag('si') if read.has_tag('si') else None
            if ss_tag and si_tag:
                start_raw, end_raw, start_kmer, end_kmer = map(int, si_tag.split(","))
                query = read.query_sequence[start_kmer:end_kmer]
                #print("\n \n", query)
                nucleotides = [nucleotide_mapping[nuc] for nuc in query if nuc in nucleotide_mapping]
                #print("\n \n",nucleotides)
                sam_data[read_id] = {
                    'nucleotides': nucleotides,
                    'ss': ss_tag,
                    'start_raw': start_raw,
                    'end_raw': end_raw
                }
            

    return sam_data
'''
def parse_sam_file(sam_file_path):
    nucleotide_mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    sam_data = {}

    with pysam.AlignmentFile(sam_file_path, "r") as samfile:
        for read in samfile:
            read_id = read.query_name
            # Extract ss and si tags (signal start and signal end)
            ss_tag = read.get_tag('ss') if read.has_tag('ss') else None
            si_tag = read.get_tag('si') if read.has_tag('si') else None
            
            if ss_tag and si_tag:
                # Check if the si_tag is correctly formatted
                si_parts = si_tag.split(",")
                ss_tag = [int(num) for num in ss_tag.split(',') if num.isdigit()]
                if len(si_parts) != 4:
                    print(f"Warning: Unexpected format for si_tag in read {read_id}: {si_tag}")
                    continue
                
                start_raw, end_raw, start_kmer, end_kmer = map(int, si_parts)
                
                # Extract the nucleotide sequence from the read
                query = read.query_sequence
                
                # Ensure the query is sliced correctly
                #query_segment = query[start_kmer:end_kmer]
                
                # Convert the nucleotide sequence to numerical representation
                nucleotides = [nucleotide_mapping[nuc] for nuc in query if nuc in nucleotide_mapping]
                
                # Only add to sam_data if nucleotides are found
                if nucleotides:
                    sam_data[read_id] = {
                        'nucleotides': nucleotides,
                        'ss': ss_tag,
                        'start_raw': start_raw,
                        'end_raw': end_raw
                    }
                else:
                    print(f"Warning: No valid nucleotides found in read {read_id}.")
    
    return sam_data
def length_of_read(signal_len, ss_values, pos, ss_len):
    read_length = 0
    
    while signal_len > 0 and pos < ss_len:
        signal_len -= ss_values[pos]
        if(signal_len <= 0):
            return read_length
        
        read_length += 1
        pos += 1
    return read_length

def generate_overlapping_segments(signal, nucleotides, ss_values, fixed_length=FIXED_LENGTH, overlap=OVERLAP):
    segments = []
    step = fixed_length - overlap
    nuc_start = 0
    ss_len = len(ss_values)
    # Ensure that the signal is at least as long as fixed_length
    if len(signal) < fixed_length:
        return segments  # No segments can be generated if the signal is too short

    for i in range(0, len(signal) - fixed_length + 1, step):
        start = i
        end = start + fixed_length
        
        segment_signal = signal[start:end]
        
        # Calculate the nucleotide end index based on the segment length
        nuc_end = nuc_start + length_of_read(len(segment_signal), ss_values, nuc_start, ss_len)

        # Ensure that the nucleotide segment does not exceed available nucleotides
        if nuc_end > len(nucleotides):
            nuc_end = len(nucleotides)  # Cap to available nucleotide length
        
        segment_nucleotides = nucleotides[nuc_start:nuc_end]
        segment_len = len(segment_nucleotides)

        # Only proceed if segment length is valid
        if len(segment_signal) == fixed_length and segment_len > 100:
            segments.append((segment_signal, segment_nucleotides))
        
        # Update nuc_start for the next segment
        nuc_start = nuc_end
        
    return segments
# Process each read, generate overlapping segments, and save as .npy
def process_reads(slow5_data, sam_data, output_dir):
    
    all_signals = []
    all_nucleotides = []
    all_lengths = []

    for read_id, signal in slow5_data.items():
        if read_id in sam_data:
            sam_line = sam_data[read_id]
            segments = generate_overlapping_segments(signal, sam_line['nucleotides'], sam_line['ss'])

            # Collect signals and nucleotides from segments
            for seg_signal, seg_nucleotides in segments:
                all_signals.append(seg_signal)  # Append signal segment
                all_nucleotides.append(seg_nucleotides)  # Append nucleotide segment
                all_lengths.append(len(seg_nucleotides))  # Append the length of the nucleotide segment

    # Prepare numpy arrays for storage
    chunks = np.array(all_signals, dtype=np.float16)
                
    max_length = max(all_lengths)  # Determine the maximum length for targets_
    targets_ = np.zeros((chunks.shape[0], max_length), dtype=np.uint8)
    
    # Fill in the targets array
    for idx, target in enumerate(all_nucleotides): 
        targets_[idx, :len(target)] = target  # Now target should be a list/array

    lengths = np.array(all_lengths, dtype=np.uint16)
    
    # Save the combined signals and nucleotides to .npy files
    np.save(f"{output_dir}/chunks.npy", chunks)  # Save all signals
    np.save(f"{output_dir}/references.npy", targets_)  # Save all nucleotides
    np.save(f"{output_dir}/reference_lengths.npy", lengths)  # Save all lengths

    sys.stderr.write(f"> written ctc training data to {output_dir}\n")
    sys.stderr.write("  - chunks.npy with shape (%s)\n" % ','.join(map(str, chunks.shape)))
    sys.stderr.write("  - references.npy with shape (%s)\n" % ','.join(map(str, targets_.shape)))
    sys.stderr.write("  - reference_lengths.npy shape (%s)\n" % ','.join(map(str, lengths.shape)))


# Main script
if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write("Usage: python preprocess.py <slow5_file> <sam_file> <output_directory>\n")
        sys.exit(1)

    slow5_file_path = sys.argv[1]
    sam_file_path = sys.argv[2]
    output_dir = sys.argv[3]

    # Load data
    slow5_data = load_slow5_data(slow5_file_path)
    sam_data = parse_sam_file(sam_file_path)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process and save overlapping segments for each read
    process_reads(slow5_data, sam_data, output_dir)
