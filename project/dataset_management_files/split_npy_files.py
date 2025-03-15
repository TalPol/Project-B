import numpy as np
import os
import argparse

def split_npy_file(input_file, output_dir, val_percent, test_percent, seed=None):
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Use memory-mapped array to avoid loading the entire file into RAM
    data = np.load(input_file, mmap_mode="r")
    total_examples = len(data)

    # Shuffle the indices
    indices = np.arange(total_examples)
    np.random.shuffle(indices)

    # Calculate split sizes
    val_size = int(total_examples * val_percent / 100)
    test_size = int(total_examples * test_percent / 100)
    train_size = total_examples - val_size - test_size

    # Split the indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Save the splits using memory-efficient indexing
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Save the subsets in the appropriate directories
    np.save(os.path.join(train_dir, f"{base_name}.npy"), data[train_indices])
    np.save(os.path.join(val_dir, f"{base_name}.npy"), data[val_indices])
    np.save(os.path.join(test_dir, f"{base_name}.npy"), data[test_indices])
    
    print(f"Processed {input_file}:")
    print(f"  Training examples: {train_size}")
    print(f"  Validation examples: {val_size}")
    print(f"  Test examples: {test_size}")


def main():
    parser = argparse.ArgumentParser(description="Split .npy files into training, validation, and test sets without high memory usage.")
    parser.add_argument("input_dir", help="Directory containing .npy files to process.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the split files.")
    parser.add_argument("--val_percent", type=float, required=True, help="Percentage of data to use for validation.")
    parser.add_argument("--test_percent", type=float, required=True, help="Percentage of data to use for testing.")
    parser.add_argument("--seed", type=int, default=24, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all .npy files in input directory
    npy_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(".npy")]

    if not npy_files:
        print(f"No .npy files found in {args.input_dir}. Exiting.")
        return

    # Process each file one by one to avoid high memory usage
    for input_file in npy_files:
        split_npy_file(input_file, args.output_dir, args.val_percent, args.test_percent, seed=args.seed)


if __name__ == "__main__":
    main()
