import time
import numpy as np
from Bio import SeqIO
import pysam
from pysam import AlignmentFile
import parasail
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from itertools import starmap
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import subprocess

from bonito.data import load_numpy, load_script
from bonito.util import accuracy, poa, decode_ref, half_supported
from bonito.util import init, load_model, permute, parasail_to_sam

def accuracy(ref, seq, balanced=False, min_coverage=0.0):
    alignment = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)
    counts = defaultdict(int)

    q_coverage = len(alignment.traceback.query) / len(seq)
    r_coverage = len(alignment.traceback.ref) / len(ref)

    if r_coverage < min_coverage:
        return 0.0

    _, cigar = parasail_to_sam(alignment, seq)

    for count, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar):
        counts[op] += int(count)

    if balanced:
        accuracy = (counts['='] - counts['I']) / (counts['='] + counts['X'] + counts['D'])
    else:
        accuracy = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])
    return accuracy * 100

# Parse FASTA file for references
def parse_fasta(fasta_file):
    return {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")}

# Parse SAM file for reads and basecalls
def parse_sam(sam_file):
    reads = []
    with AlignmentFile(sam_file, "r") as sam:
        for read in sam.fetch(until_eof=True):
            if not read.is_unmapped:
                reads.append((read.query_name, read.query_sequence))
    return reads

def basecall_with_bonito(model_path, input_dir, output_sam, reference, batchsize=32, device="cuda"):
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
    # Redirect output to a SAM file
    with open(output_sam, "w") as sam_file:
        subprocess.run(bonito_command, stdout=sam_file, check=True)
    duration = time.time() - start_time
    print(f"Basecalling completed in {duration:.2f} seconds.")
    return duration

def main(args):
    print("* loading data")
    references = pysam.FastaFile(args.reference)
    #references = parse_fasta(args.reference)
    init(args.seed, args.device)
    output_dir = args.output_dir
    input_dir = args.directory
    alignment_sam = output_dir / "alignment.sam"

    accuracy_with_cov = lambda ref, seq: accuracy(ref, seq, min_coverage=0.0)
    duration = basecall_with_bonito(
        model_path=args.model_directory,
        input_dir=input_dir,
        output_sam=alignment_sam,
        reference=args.reference,
    )
    #reads = parse_sam(alignment_sam)
    reads = pysam.AlignmentFile(alignment_sam, "rb")

    seqs = []

    print("* loading model")
    model = load_model(args.model_directory, args.device)
    mean = model.config.get("standardisation", {}).get("mean", 0.0)
    stdev = model.config.get("standardisation", {}).get("stdev", 1.0)
    print(f"* * signal standardisation params: mean={mean}, stdev={stdev}")

    print("* calling")

    accuracies = []
    with torch.no_grad():
        for read in reads.fetch():
            if read.is_unmapped:
                #print(f"Read name {read_name} not in references.")
                continue
            read_seq = read.query_sequence
            #ref_seq = references[read_name]
            ref_seq = references.fetch(read.reference_name, read.reference_start, read.reference_end)
            seqs.append(read_seq)
            accuracies.append(accuracy_with_cov(ref_seq, read_seq) if len(read_seq) else 0.0)


    print("* mean      %.2f%%" % np.mean(accuracies))
    print("* median    %.2f%%" % np.median(accuracies))
    print("* time      %.2f" % duration)
    # print("* samples/s %.2E" % (len(seqs) / duration))

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=9, type=int)
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--chunks", default=1000, type=int)
    parser.add_argument("--batchsize", default=96, type=int)
    parser.add_argument("--beamsize", default=5, type=int)
    parser.add_argument("--poa", action="store_true", default=False)
    parser.add_argument("--min-coverage", default=0.5, type=float)
    return parser

if __name__ == "__main__":
    print("hi")
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(
        title="subcommands", description="valid commands",
        help="additional help", dest="command"
    )
    subparsers.required = True

    # Add the "run" subcommand
    run_parser = subparsers.add_parser(
        "run", help="Run basecalling and evaluation",
        parents=[argparser()]
    )
    run_parser.set_defaults(func=main)

    args = parser.parse_args()
    args.func(args)