#!/usr/bin/env python3
"""
Copyright 2019 Ryan Wick (rrwick@gmail.com)
https://github.com/rrwick/Basecalling-comparison

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version. This program is distributed in the hope that it
will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should
have received a copy of the GNU General Public License along with this program. If not, see
<http://www.gnu.org/licenses/>.

This script reads a tsv file (for either read or assembly data) and returns the median identity.
It can optionally take a total number of sequences, which will make it add zeros to get to that
number (useful for basecallers that didn't basecall all of the reads).
"""

import statistics
import sys
import numpy as np
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

def basecall_with_bonito(model_path, input_dir, output_sam, reference, batchsize=32, device="cuda"):
    """
    Perform basecalling using a Bonito-trained model.
    """
    bonito_command = [
        "bonito", "basecaller", str(model_path), str(input_dir),
        "--device", device,
        "--batchsize", str(batchsize),
        "--reference", str(reference),
        "--recursive",
    ]
    
    print("Running command:", " ".join(bonito_command))
    start_time = time.perf_counter()
    # Redirect output to a SAM file
    with open(output_sam, "w") as sam_file:
        subprocess.run(bonito_command, stdout=sam_file, check=True)
    duration = time.perf_counter() - start_time
    #print(f"Basecalling completed in {duration:.2f} seconds.")
    return duration

def main(args):
    #references = pysam.FastaFile(args.reference)
    #references = parse_fasta(args.reference)
    init(args.seed, args.device)
    output_dir = args.output_dir
    input_dir = args.directory
    alignment_sam = output_dir / "basecalls.sam"
    alignment_file = output_dir / "basecalls_summary.tsv"
    
    duration = basecall_with_bonito(
        model_path=args.model_directory,
        input_dir=input_dir,
        output_sam=alignment_sam,
        reference=args.reference,
    )
    #reads = parse_sam(alignment_sam)
    #reads = pysam.AlignmentFile(alignment_sam, "rb")
    #seqs = []
    identities = []
    identities_with_unmapped = []
    unmapped = 0
    with open(alignment_file, 'rt') as data_file:
        for line in data_file:
            parts = line.strip().split()
            if parts[0] == 'filename' or parts[0] == '-':
                continue
            identities_with_unmapped.append(float(parts[-1]))
            if float(parts[-1]) == 0.0:
                unmapped += 1
                continue

            #print(parts[10])
            #identity = np.exp(np.log(10) * (-float( parts[-1]) / 10))
            #print(parts[10])
            #identities.append(1 - identity)
            identities.append(float(parts[-1]))
    sys.stderr.write("> mean accuracy: %s\n" % np.mean(identities))
    sys.stderr.write("> median accuracy: %s\n" % np.median(identities))
    sys.stderr.write("> unmapped reads: %s\n" % unmapped)
    sys.stderr.write("> mean accuracy with unmapped reads: %s\n" % np.mean(identities_with_unmapped))
    sys.stderr.write("> median accuracy with unmapped reads: %s\n" % np.median(identities_with_unmapped))
    # print("* mean       %.5f%%" % np.mean(identities))
    # print("* median     %.5f%%" % np.median(identities))
    # print("* unmapped   ", unmapped)
    


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

'''
identities = []
unmapped = 0
with open(sys.argv[1], 'rt') as data_file:
    for line in data_file:
        parts = line.strip().split()
        if parts[0] == 'filename' or parts[0] == '-':
            continue
        if float(parts[-1]) == 0.0:
            unmapped += 1
            continue

        #print(parts[10])
        #identity = np.exp(np.log(10) * (-float( parts[-1]) / 10))
        #print(parts[10])
        #identities.append(1 - identity)
        identities.append(float(parts[-1]))


print(np.mean(identities))
print("unmapped reads:", unmapped)
'''