import h5py

with h5py.File('your_file.fast5', 'r') as f:
    reads = [key for key in f.keys() if key.startswith('read_')]
    if len(reads) == 1:
        print("Single-read FAST5 file")
    else:
        print(f"Multi-read FAST5 file with {len(reads)} reads")
