'''
import numpy as np
import sys
import glob

base_data_dir = sys.argv[1]

## combine the chunk arrays
chunks_filename = 'chunks'
chunks_pattern = f"{base_data_dir}/*/{chunks_filename}.npy"
chunks_array = np.vstack( [np.load(file_path) for file_path in glob.glob(chunks_pattern)] )

np.save(f"{base_data_dir}/{chunks_filename}.npy",chunks_array)

## combine the reference_lengths arrays
ref_len_filename = 'reference_lengths'
ref_len_pattern = f"{base_data_dir}/*/{ref_len_filename}.npy"
ref_len_array = np.concatenate( [np.load(file_path) for file_path in glob.glob(ref_len_pattern)] )
np.save(f"{base_data_dir}/{ref_len_filename}.npy",ref_len_array)

## combine the reference arrays
max_ref_len = int(np.max(ref_len_array))
ref_filename = 'references'
ref_pattern = f"{base_data_dir}/*/{ref_filename}.npy"
old_ref_array = [np.load(file_path) for file_path in glob.glob(ref_pattern)]
max_ref_len = int(np.max([x.shape[1] for x in old_ref_array]))

ref_array = []
for old_array in old_ref_array:
  # zerofill the new array to 'max_ref_len' in the second dimension
  new_array = np.zeros((old_array.shape[0],max_ref_len),dtype=old_array.dtype)
  new_array[:, :old_array.shape[1]] = old_array
  ref_array.append(new_array)
ref_array = np.vstack( ref_array )
np.save(f"{base_data_dir}/{ref_filename}.npy",ref_array)
'''
import numpy as np
import sys
import glob

base_data_dir = sys.argv[1]

## Step 1: Calculate mean and standard deviation for chunks in a memory-efficient way
chunks_filename = 'chunks'
chunks_pattern = f"{base_data_dir}/*/{chunks_filename}.npy"

# First pass: Calculate global mean and variance in a way that avoids overflow
mean_sum = 0.0
sq_sum = 0.0
total_count = 0

for file_path in glob.glob(chunks_pattern):
    data = np.load(file_path, mmap_mode='r')
    mean_sum += np.sum(data, dtype=np.float64)  # Use float64 to avoid overflow
    sq_sum += np.sum(np.square(data, dtype=np.float64))  # Square with float64
    total_count += data.size

# Calculate the global mean and standard deviation
global_mean = mean_sum / total_count
global_std = np.sqrt(sq_sum / total_count - global_mean ** 2)
print("\nGlobal Mean is:", global_mean)
print("\nGlobal Std Dev is:", global_std)

# Second pass: Standardize each chunk file using the calculated mean and std
chunk_files = glob.glob(chunks_pattern)
chunk_shape = (sum(np.load(file, mmap_mode='r').shape[0] for file in chunk_files),) + np.load(chunk_files[0], mmap_mode='r').shape[1:]

# Initialize a memory-mapped output array for standardized chunks
standardized_memmap = np.memmap(f"{base_data_dir}/temp_standardized_chunks.npy", dtype='float32', mode='w+', shape=chunk_shape)

# Process each chunk file and save to standardized_chunks
current_index = 0
for file_path in chunk_files:
    data = np.load(file_path, mmap_mode='r')
    standardized_memmap[current_index:current_index + data.shape[0]] = (data - global_mean) / global_std
    current_index += data.shape[0]

# Cast the memmap to float16 to save space
standardized_memmap = standardized_memmap.astype('float16')

# Flush changes and save as a standard .npy file
standardized_memmap.flush()  # Ensure data is written
np.save(f"{base_data_dir}/{chunks_filename}.npy", standardized_memmap)  # Save as float16

## Step 2: Combine the reference_lengths arrays
ref_len_filename = 'reference_lengths'
ref_len_pattern = f"{base_data_dir}/*/{ref_len_filename}.npy"
ref_len_array = np.concatenate([np.load(file_path) for file_path in glob.glob(ref_len_pattern)])
np.save(f"{base_data_dir}/{ref_len_filename}.npy", ref_len_array)

## Step 3: Combine and zero-pad the reference arrays to max length
ref_filename = 'references'
ref_pattern = f"{base_data_dir}/*/{ref_filename}.npy"
old_ref_array = [np.load(file_path) for file_path in glob.glob(ref_pattern)]
max_ref_len = int(np.max([x.shape[1] for x in old_ref_array]))

ref_array = []
for old_array in old_ref_array:
    # Zero-fill each array to max_ref_len in the second dimension
    new_array = np.zeros((old_array.shape[0], max_ref_len), dtype=old_array.dtype)
    new_array[:, :old_array.shape[1]] = old_array
    ref_array.append(new_array)
ref_array = np.vstack(ref_array)
np.save(f"{base_data_dir}/{ref_filename}.npy", ref_array)

# Output shapes
sys.stderr.write("  - chunks.npy with shape (%s)\n" % ','.join(map(str, standardized_memmap.shape)))
sys.stderr.write("  - references.npy with shape (%s)\n" % ','.join(map(str, ref_array.shape)))
sys.stderr.write("  - reference_lengths.npy shape (%s)\n" % ','.join(map(str, ref_len_array.shape)))
