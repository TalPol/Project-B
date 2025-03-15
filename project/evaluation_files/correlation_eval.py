'''
File logic:
From squigulator/test_set/results/alignment_summary.tsv
extract the qscore_mean, alignment accuracy, line length

according to these parameters calculate the following:
qscore_mean of all the file, version 1- give each line a weight according to its' length and calculate the mean_qscore_mean

qscore_mean of all the file, version 2- give each line equal weight, calculate the median q_score_mean of the file

mean_alignment_accuracy, version 1- give each line a weight according to its' length and calculate the mean_we_alignment_accuracy

mean_alignment_accuracy, version 2- give each line equal weight according to its' length and calculate the mean_eq_alignment_accuracy

q_score_auto_correlation- the auto-correlation of the q_score_mean, this function should tell us about credibility of q_scoe results,
it can also tell us about the consistency of q_score.


alignment_score_auto_correlation- the auto-correlation of the alignment score, this function should tell us about credibility of the alingment score results,
it can also tell us about the consistency of prediction of the alignment. 

cross_correlation_q_score_alignment_score- the cross correlation of the q_score and the alignment score, also related to credibility

'''
import pandas as pd
import numpy as np
from scipy.signal import correlate, welch
import matplotlib.pyplot as plt


# Function to compute auto-correlation
def auto_correlation(data):
    # Normalize the data before computing the autocorrelation
    data = data - np.mean(data)
    correlation = correlate(data, data, mode='full')
    return correlation[correlation.size // 2:]  # Get the positive part

# Function to compute cross-correlation
def cross_correlation(data1, data2):
    # Normalize both datasets before computing the cross-correlation
    data1 = data1 - np.mean(data1)
    data2 = data2 - np.mean(data2)
    correlation = correlate(data1, data2, mode='full')
    return correlation[correlation.size // 2:]  # Get the positive part

# Function to compute the Power Spectral Density (PSD) from the auto-correlation
# fs MUST be calculated according to the Nyquist Theorom, according to the average time it takes to produce a line
# Calculating fs according to the longest time it takes  
def compute_psd_from_corr(correlation, fs=1000.0):
    # Compute the Fourier Transform of the correlation (auto-correlation)
    psd = np.abs(np.fft.fft(correlation))**2
    
    # Compute the frequencies corresponding to the PSD
    n = len(correlation)
    freq = np.fft.fftfreq(n, d=fs)  # fs is the sampling frequency, default to 1.0 for unit time steps
    
    # Only keep the positive frequencies (symmetry in FFT)
    psd = psd[:n // 2]
    freq = freq[:n // 2]
    
    return freq, psd

# Function to read and validate the input file
def read_data(file_path):
    try:
        # Read the TSV file into a pandas DataFrame
        df = pd.read_csv(file_path, sep='\t')
        
        # Check for required columns
        required_columns = ['mean_qscore_template', 'alignment_accuracy', 'sequence_length_template']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Drop rows with NaN values in relevant columns
        df = df.dropna(subset=required_columns)
        
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None

# Read the data from the TSV file
file_path = 'squigulator/test_set/results/alignment_summary.tsv'
df = read_data(file_path)

# Check if the data was successfully loaded
if df is not None:
    # Extract the necessary columns from the DataFrame
    qscore_mean = df['mean_qscore_template']
    alignment_accuracy = df['alignment_accuracy']
    line_length = df['sequence_length_template']

    # 1. Weighted qscore_mean (version 1)
    weighted_qscore_mean = np.sum(qscore_mean * line_length) / np.sum(line_length)

    # 2. Median qscore_mean (version 2)
    median_qscore_mean = np.median(qscore_mean)

    # 3. Weighted alignment accuracy (version 1)
    weighted_alignment_accuracy = np.sum(alignment_accuracy * line_length) / np.sum(line_length)

    # 4. Median alignment accuracy (version 2)
    median_alignment_accuracy = np.median(alignment_accuracy)

    # 5. Auto-correlation of qscore_mean
    qscore_auto_corr = auto_correlation(qscore_mean)
    qscore_auto_corr_norm = qscore_auto_corr/qscore_auto_corr[0]
    
    # 6. Auto-correlation of alignment_accuracy
    alignment_auto_corr = auto_correlation(alignment_accuracy)
    alignment_auto_corr_norm = alignment_auto_corr/alignment_auto_corr[0]

    # 7. Cross-correlation between qscore_mean and alignment_accuracy
    cross_corr_qscore_alignment = cross_correlation(qscore_mean, alignment_accuracy)
    cross_corr_qscore_alignment_norm=cross_corr_qscore_alignment/cross_corr_qscore_alignment[0]
    
    # Print the results
    print(f'Weighted Mean Q-score: {weighted_qscore_mean}')
    print(f'Median Q-score: {median_qscore_mean}')
    print(f'Weighted Mean Alignment Accuracy: {weighted_alignment_accuracy}')
    print(f'Median Alignment Accuracy: {median_alignment_accuracy}')

    # Plotting auto-correlations (optional visualization)
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.plot(qscore_auto_corr)
    plt.title('Auto-correlation of Q-score Mean')
    plt.grid()
    
    plt.subplot(2, 2, 2)
    plt.plot(alignment_auto_corr)
    plt.title('Auto-correlation of Alignment Accuracy')
    plt.grid()
    
    plt.subplot(2, 2, 3)
    plt.plot(cross_corr_qscore_alignment)
    plt.title('Cross-correlation of Q-score Mean and Alignment Accuracy')
    plt.grid()
    
    plt.tight_layout()
    plt.savefig('autocorrelations.png')
    
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.plot(qscore_auto_corr_norm)
    plt.title('Auto-correlation of Q-score Mean- Normalized')
    plt.grid()
    
    plt.subplot(2, 2, 2)
    plt.plot(alignment_auto_corr_norm)
    plt.title('Auto-correlation of Alignment Accuracy- Normalized')
    plt.grid()
    
    plt.subplot(2, 2, 3)
    plt.plot(cross_corr_qscore_alignment_norm)
    plt.title('Cross-correlation of Q-score Mean and Alignment Accuracy- Normalized')
    plt.grid()
    
    plt.tight_layout()
    plt.savefig('autocorrelations_normalized.png')
    

    # Optional: Plot histograms for Q-score and Alignment Accuracy
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    df['mean_qscore_template'].hist(bins=30)
    plt.title('Q-score Distribution')
    plt.xlabel('Q-score')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    df['alignment_accuracy'].hist(bins=30)
    plt.title('Alignment Accuracy Distribution')
    plt.xlabel('Alignment Accuracy')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('histograms.png')

    
    # Calculate the PSD for qscore_mean and alignment_accuracy
    freq_qscore, psd_qscore = compute_psd_from_corr(qscore_auto_corr)
    freq_alignment, psd_alignment = compute_psd_from_corr(alignment_auto_corr)
    freq_alignment_and_qscore, psd_cross_qscore_alignment= compute_psd_from_corr(cross_corr_qscore_alignment)

    # Plotting PSD
    plt.figure(figsize=(10, 6))

    # Plot the PSD of Q-score Mean
    plt.subplot(2, 2, 1)
    plt.semilogy(freq_qscore, psd_qscore)
    plt.title('Power Spectral Density of Q-score Mean')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid()

    # Plot the PSD of Alignment Accuracy
    plt.subplot(2, 2, 2)
    plt.semilogy(freq_alignment, psd_alignment)
    plt.title('Power Spectral Density of Alignment Accuracy')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid()
    
    plt.subplot(2, 2, 3)
    plt.semilogy(freq_alignment_and_qscore, psd_cross_qscore_alignment)
    plt.title('Power Spectral Density of Alignment Accuracy')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid()

    plt.tight_layout()
    plt.savefig('PSD.png')
