import os
import subprocess
import argparse
from time import time

def run_command(command, capture_output=False):
    """Run a shell command and optionally capture output."""
    result = subprocess.run(command, shell=True, text=True, capture_output=capture_output)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}\n{result.stderr}")
    return result.stdout.strip() if capture_output else None

def measure_throughput(output_file, elapsed_time):
    """Calculate throughput as bases per second."""
    with open(output_file, 'r') as f:
        total_bases = sum(len(line.strip()) for line in f if not line.startswith('>'))
    throughput = total_bases / elapsed_time
    return throughput, total_bases

def evaluate_alignment(sam_file):
    """Evaluate alignment accuracy using paftools.js or other tools."""
    command = f"paftools.js mapeval {sam_file}"
    output = run_command(command, capture_output=True)
    return output  # You can parse this further for specific metrics if needed

def main(args):
    # Measure model size
    model_size = os.path.getsize(args.model) / (1024 * 1024)  # Size in MB

    # Basecall with Dorado and measure time
    print("Running Dorado basecaller...")
    start_time = time()
    output_fasta = os.path.join(args.output_dir, "basecalls.fasta")
    run_command(f"dorado basecall --model {args.model} --input {args.input} --output {output_fasta}")
    elapsed_time = time() - start_time
    print(f"Basecalling completed in {elapsed_time:.2f} seconds.")

    # Measure throughput
    throughput, total_bases = measure_throughput(output_fasta, elapsed_time)
    print(f"Throughput: {throughput:.2f} bases per second (total bases: {total_bases}).")

    # Align basecalls to reference
    print("Aligning basecalls to reference...")
    sam_file = os.path.join(args.output_dir, "aligned.sam")
    run_command(f"minimap2 -ax map-ont {args.reference} {output_fasta} > {sam_file}")

    # Evaluate read accuracy
    print("Evaluating alignment accuracy...")
    accuracy_metrics = evaluate_alignment(sam_file)
    print(f"Accuracy metrics:\n{accuracy_metrics}")

    # Summary
    print("\nEvaluation Summary:")
    print(f"Model Size: {model_size:.2f} MB")
    print(f"Throughput: {throughput:.2f} bases per second")
    print(f"Total Bases: {total_bases}")
    print(f"Accuracy Metrics: {accuracy_metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dorado basecaller performance.")
    parser.add_argument("--model", required=True, help="Path to the Dorado model.")
    parser.add_argument("--input", required=True, help="Directory containing input .pod5 files.")
    parser.add_argument("--reference", required=True, help="Path to the reference .fasta file.")
    parser.add_argument("--output_dir", required=True, help="Directory to store output files.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}")
