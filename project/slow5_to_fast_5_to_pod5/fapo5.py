import os
import sys
import h5py
from pod5 import convert_fast5_to_pod5

#Arguments:
input_path = sys.argv[1]
output_path = sys.argv[2]

def main():
    convert_fast5_to_pod5(input_path,output_path)
