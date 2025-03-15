import os
import time
import subprocess
import torch
import pysam
from pathlib import Path
from Bio import SeqIO
import re

def preprocess_sam_for_paftools(sam_file_path, processed_sam_file_path):
    with open(sam_file_path, "r") as infile, open(processed_sam_file_path, "w") as outfile:
        for line in infile:
            if line.startswith("@"):  # Skip header lines
                outfile.write(line)
            else:
                # Modify read name by removing the UUID part (if it exists)
                fields = line.split("\t")
                read_name = fields[0]
                # Remove UUID from read name (assuming UUID format like 'read_b5268d0d-f801-4d53-bdf9-e1e28b933ffd')
                new_read_name = re.sub(r"read_[a-f0-9\-]+", "read", read_name)
                fields[0] = new_read_name
                outfile.write("\t".join(fields) + "\n")

    print(f"Modified SAM file saved as {processed_sam_file_path}")

    
def basecall_with_bonito(model_path, input_dir, output_fastq, reference, batchsize=32, device="cuda"):
    """
    Perform basecalling using a Bonito-trained model.
    """
    print("Starting basecalling with Bonito...")
    bonito_command = [
        "bonito", "basecaller", model_path, input_dir,
        "--device", device,
        "--batchsize", str(batchsize),
        "--reference", reference,
    ]
    start_time = time.time()
    # Redirect output to a FASTA file
    with open(output_fastq, "w") as fastq_file:
        subprocess.run(bonito_command, stdout=fastq_file, check=True)
    duration = time.time() - start_time
    print(f"Basecalling completed in {duration:.2f} seconds.")
    return duration

def align_basecalls_to_reference(basecalls, reference, sam_output):
    """
    Align basecalled reads to the reference using Minimap2.
    """
    print("Aligning basecalls to reference...")
    minimap_command = [
        "minimap2", "-ax", "map-ont", reference, basecalls
    ]
    with open(sam_output, "w") as sam_file:
        subprocess.run(minimap_command, stdout=sam_file, check=True)
    print("Alignment completed.")

'''
def calculate_accuracy(sam_file):
    """
    Calculate read accuracy from the SAM file using Paftools.
    """
    print("Calculating read accuracy...")
    paftools_command = ["./minimap2-2.28_x64-linux/paftools.js", "mapeval", sam_file]
    result = subprocess.run(
    paftools_command, capture_output=True, text=True
    )
    #print("Standard Output:\n", result.stdout)
    print("Error Output:\n", result.stderr)
    for line in result.stdout.splitlines():
        print(line)  # Outputs metrics
    return result.stdout
'''
def calculate_accuracy(sam_file, reference_file):
    """
    Calculates basecalling accuracy by comparing aligned reads to the reference genome.
    
    Args:
        sam_file (str): Path to the SAM or BAM file.
        reference_file (str): Path to the reference genome in FASTA format.
        
    Returns:
        dict: Accuracy metrics including matches, mismatches, insertions, deletions, and overall accuracy.
    """
    # Open the SAM/BAM file
    sam = pysam.AlignmentFile(sam_file, "rb")
    ref = pysam.FastaFile(reference_file)
    
    total_matches = 0
    total_mismatches = 0
    total_insertions = 0
    total_deletions = 0
    total_bases = 0
    total_unmapped = 0
    for read in sam.fetch():
        if read.is_unmapped:  # Skip unmapped reads
            total_unmapped += 1
            continue

        query_seq = read.query_sequence
        ref_seq = ref.fetch(read.reference_name, read.reference_start, read.reference_end)

        # Parse the CIGAR string to determine matches, mismatches, insertions, and deletions
        ref_pos = 0  # Position in reference sequence
        query_pos = 0  # Position in query sequence

        for cigar_op, length in read.cigartuples:
            if cigar_op == 0:  # Match or mismatch
                query_segment = query_seq[query_pos:query_pos + length]
                ref_segment = ref_seq[ref_pos:ref_pos + length]
                for q, r in zip(query_segment, ref_segment):
                    if q == r:
                        total_matches += 1
                    else:
                        total_mismatches += 1
                query_pos += length
                ref_pos += length
            elif cigar_op == 1:  # Insertion in query
                total_insertions += length
                query_pos += length
            elif cigar_op == 2:  # Deletion in reference
                total_deletions += length
                ref_pos += length
            elif cigar_op in (3, 4, 5):  # Clipped regions or skipped reference bases
                continue

    # Calculate total bases and accuracy
    total_bases = total_matches + total_mismatches + total_insertions + total_deletions
    accuracy = total_matches / total_bases if total_bases > 0 else 0.0
    print("total unmapped: ", total_unmapped)
    '''
    return {
        "matches": total_matches,
        "mismatches": total_mismatches,
        "insertions": total_insertions,
        "deletions": total_deletions,
        "total_bases": total_bases,
        "accuracy": accuracy,
    }
    '''
    return accuracy

def measure_model_size(model_path):
    """
    Measure the model size in megabytes.
    """
    weights_file = Path(model_path) / 'weights_1.tar'
    if not Path(weights_file).exists():
        print(f"Error: Model file at {model_path} does not exist.")
        return 0
    model_size = Path(weights_file).stat().st_size / (1024 * 1024)
    print(f"Model size: {model_size:.2f} MB")
    return model_size

def evaluate_bonito_model(model_path, data_dir, reference, output_dir):
    """
    Evaluate a Bonito model on accuracy, throughput, and size.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    basecalls_fastq = output_dir / "basecalls.fastq"
    alignment_sam = output_dir / "alignment.sam"
    #processed_sam_file = output_dir / "processed_alignment.sam"

    # Basecalling
    throughput = basecall_with_bonito(
        model_path=model_path,
        input_dir=data_dir,
        output_fastq=alignment_sam,
        reference=reference,
    )
    
    # Align basecalls
    '''
    align_basecalls_to_reference(
        basecalls=basecalls_fastq,
        reference=reference,
        sam_output=alignment_sam,
    )
    '''
    # Preprocess SAM for Paftools
    #preprocess_sam_for_paftools(alignment_sam, processed_sam_file)

    # Calculate accuracy
    accuracy = calculate_accuracy(alignment_sam, reference)

    # Measure model size
    model_size = measure_model_size(model_path=model_path)

    return {
        "accuracy_metrics": accuracy,
        "throughput_seconds": throughput,
        "model_size_mb": model_size,
    }

if __name__ == "__main__":
    # Define paths and parameters
    model_path = "./squigulator/models/bonito_hac_lr_2.5e-4"  # Path to Bonito model
    #model_path = "./bonito/bonito/models/dna_r10.4.1_e8.2_400bps_fast@v4.3.0"
    data_dir = "./squigulator/dataset/train/pod_files/ATCC_BAA-679__202309/batch_1"  # Directory containing .pod5 files
    reference = "./squigulator/dataset/train/fasta/ATCC_BAA-679__202309/reference.fasta"     # Path to reference fasta file
    output_dir = "./squigulator/test_set/results/" # Directory for results

    # Evaluate the model
    results = evaluate_bonito_model(model_path, data_dir, reference, output_dir)
    print("\nEvaluation Results:")
    print(f"Accuracy Metrics: {results['accuracy_metrics']}")
    print(f"Throughput: {results['throughput_seconds']:.2f} seconds")
    print(f"Model Size: {results['model_size_mb']:.2f} MB")
