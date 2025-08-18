
#%% 
########################################<<<  Load Libraries  >>>######################################## 

from Load.LoadData import DataProcessor, SkinTempProcessor, DataNormalizer 
from Utils.func import SignalProcessor 
from PreProcessing import Resp_SQA, EDA_SQA
import os
import matplotlib.pyplot as plt
from scipy import signal
import neurokit2 as nk
import numpy as np 
import tensorflow as tf
import pandas as pd
import heartpy as hp
import flirt
from datetime import datetime

# For IKS
from scipy.stats import ks_2samp
from IKS.IKS import IKS

# The following part is specific to Jupyter notebooks and can be commented out if the script is not being run in a Jupyter environment:

# Enable interactive plots in Jupyter notebooks
%matplotlib widget
# Load the autoreload extension which reloads modules before executing code
%load_ext autoreload
# Automatically reload all modules (except those excluded by %aimport) every time before executing the Python code
%autoreload 2
# %%
########################################<<<  Load Data  >>>######################################## 

Subject_ID = 'P33M'   #BMES paper subject 5 is used 

# Define the file path and lines to skip
file_path = os.path.join(r'C:\Users\573751\OneDrive - TMNA\Desktop\Files\Datasets\Neq15\Impatience', Subject_ID + '.txt')
lines_to_skip = 20  # Adjust this number based on how many lines you want to skip

biodata = DataProcessor(file_path).process_file(lines_to_skip)

# Print the results
for i, length, data in biodata:
    print(f"Column {i+1}: {length} entries")
    print(f"Sample data: {data[:5]}...")  # Print first 5 entries as a sample
    print("(:Data Loading is done:)")  


#%%
# Add the Checkpoints file
checkpoints_file = r'C:\Users\573751\OneDrive - TMNA\Desktop\Python Codes\CPD\CPD_ML2\WESAD\Sensor Fusion\Checkpoints\Checkpoints_Imp.txt'
# Read the Checkpoints file
checkpoints = {}
with open(checkpoints_file, 'r') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) == 2:
            subject_id, value = parts
            checkpoints[subject_id] = value

# Get the value for the current Subject_ID
if Subject_ID in checkpoints:
    checkpoint_value = checkpoints[Subject_ID]
    print(f"Checkpoint value for {Subject_ID}: {checkpoint_value}")
else:
    print(f"No checkpoint value found for {Subject_ID}")

#%%
                                    ############# ((GT Extraction)) #############
def calculate_time_differences_for_sheet(df):
    """
    Calculate time differences in seconds relative to Data Start for a single sheet.
    
    Parameters:
    df (pandas.DataFrame): DataFrame from excel sheet
    
    Returns:
    pandas.DataFrame: DataFrame with time differences in seconds
    """
    # Create datetime objects for each row
    def create_datetime(row):
        return datetime(
            year=int(row['Year']),
            month=int(row['Month']),
            day=int(row['Day']),
            hour=int(row['Hour']) if pd.notna(row['Hour']) else 0,
            minute=int(row['Minute']) if pd.notna(row['Minute']) else 0,
            second=int(row['Second']) if pd.notna(row['Second']) else 0
        )
    
    # Calculate timestamps
    df['timestamp'] = df.apply(create_datetime, axis=1)
    # Get the Data Start time
    start_time = df.loc[df['Event'] == 'Data Start', 'timestamp'].iloc[0]
    # Calculate time differences in seconds
    df['seconds_from_start'] = (df['timestamp'] - start_time).dt.total_seconds()
    # Create a more readable output DataFrame
    result_df = df[['Event', 'seconds_from_start']].copy()
    return result_df

def process_excel_file(excel_file):
    """
    Process all sheets in an Excel file and calculate time differences.
    Parameters:
    excel_file (str): Path to the Excel file
    Returns:
    dict: Dictionary with sheet names as keys and processed DataFrames as values
    """
    # Read all sheets from the Excel file
    excel = pd.ExcelFile(excel_file)
    # Dictionary to store results for each sheet
    results = {}
    # Process each sheet
    for sheet_name in excel.sheet_names:
        try:
            # Read the sheet
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            # Process the sheet
            result_df = calculate_time_differences_for_sheet(df)
            # Store results
            results[sheet_name] = result_df
        except Exception as e:
            print(f"Error processing sheet {sheet_name}: {str(e)}")
            continue
    
    return results

# Function to process specific sheets
def process_specific_sheets(excel_file, sheet_names):
    """
    Process only specified sheets from an Excel file.
    
    Parameters:
    excel_file (str): Path to the Excel file
    sheet_names (list): List of sheet names to process
    
    Returns:
    dict: Dictionary with specified sheet names as keys and processed DataFrames as values
    """
    results = {}
    
    for sheet_name in sheet_names:
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            result_df = calculate_time_differences_for_sheet(df)
            results[sheet_name] = result_df
        except Exception as e:
            print(f"Error processing sheet {sheet_name}: {str(e)}")
            continue
    
    return results

input_file =  r"C:\Users\573751\OneDrive - TMNA\Desktop\Files\Datasets\Neq15\Sensor Fusion 2024\Time_Stamps\Experiment_3_Stamps_M.xlsx"
specific_sheets = [Subject_ID]
labels_time = process_specific_sheets(input_file, specific_sheets)
print(labels_time)

# Extract the time labels from labels_time
time_labelss = labels_time[Subject_ID]['seconds_from_start'].values
labels_textt = labels_time[Subject_ID]['Event'].values
# Create mask for non-negative values
mask = time_labelss >= 0
# Apply mask to both arrays
time_labels = time_labelss[mask][1:]
labels_text = labels_textt[mask][1:]
# Find indices where time_labels < 0
negative_indices = time_labelss < 0
# Store removed labels in missed_events
missed_events = labels_textt[negative_indices]


# %%
########################################<<<  Down Sampling  >>>######################################## 

# Define sample frequencies for biosignals
Biopac_fs = 2000
ecg_fs = 200
ppg_fs = 200
resp_fs = 100
eda_fs = 4
temp_fs = 4
# Define window size and overlap
window_size = 150  # seconds
window_step_size = 2  # second
# Define clipping value for the beginning and end of the signal
video_start_time = labels_time[Subject_ID].loc[labels_time[Subject_ID]['Event'] == 'Video Start', 'seconds_from_start'].values[0]
study_end_time = labels_time[Subject_ID].loc[labels_time[Subject_ID]['Event'] == 'Survey 3 End', 'seconds_from_start'].values[0]
CLIP_VALUE = [int(video_start_time),int(study_end_time)]  # seconds

# Extract each bio signal from data and clip the defined value from the start and end of the data
ecg_signal = biodata[2][-1][CLIP_VALUE[0]*Biopac_fs:CLIP_VALUE[1]*Biopac_fs] 
ppg_signal = biodata[1][-1][CLIP_VALUE[0]*Biopac_fs:CLIP_VALUE[1]*Biopac_fs]  
resp_signal = biodata[4][-1][CLIP_VALUE[0]*Biopac_fs:CLIP_VALUE[1]*Biopac_fs]   
resp_signal_2 = biodata[6][-1][CLIP_VALUE[0]*Biopac_fs:CLIP_VALUE[1]*Biopac_fs]   
eda_signal = biodata[3][-1][CLIP_VALUE[0]*Biopac_fs:CLIP_VALUE[1]*Biopac_fs]   
temp_signal = biodata[5][-1][CLIP_VALUE[0]*Biopac_fs:CLIP_VALUE[1]*Biopac_fs]   

downsampled_ecg = SignalProcessor.downsample_signal(ecg_signal, Biopac_fs, ecg_fs)
downsampled_ppg = SignalProcessor.downsample_signal(ppg_signal, Biopac_fs,ppg_fs)
downsampled_resp = SignalProcessor.downsample_signal(resp_signal, Biopac_fs, resp_fs)
downsampled_resp_2 = SignalProcessor.downsample_signal(resp_signal_2, Biopac_fs, resp_fs)
downsampled_eda = SignalProcessor.downsample_signal(eda_signal, Biopac_fs, eda_fs)
downsampled_temp = SignalProcessor.downsample_signal(temp_signal, Biopac_fs, temp_fs)

# Print the downsampled signals verify
print(f"{"Data is "}{len(downsampled_ppg) / ppg_fs} seconds")
print("(:Downsampling is done:)")


#%% 

########################################<<<  Apply LP/HP Filters  >>>######################################## 

ecg_f = nk.ecg_clean(downsampled_ecg, sampling_rate=ecg_fs, method="neurokit")
ppg_f = nk.ppg_clean(downsampled_ppg, sampling_rate= ppg_fs , method='nabian2018')
resp_f = nk.rsp_clean(downsampled_resp, sampling_rate=resp_fs, method="khodadad2018")
resp_f_2 = nk.rsp_clean(downsampled_resp_2, sampling_rate=resp_fs, method="khodadad2018")
eda_f = nk.eda_clean(downsampled_eda, sampling_rate=eda_fs, method='neurokit')
temp_f = SkinTempProcessor().process_skin_temp_signal(downsampled_temp)

print("(:Filtering is done:)")

#%%
########################################<<<  Some Initialization  >>>######################################## 

ts_ecg = np.arange(len(ecg_f)) / ecg_fs
ts_ppg = np.arange(len(ppg_f)) / ppg_fs
ts_resp = np.arange(len(resp_f)) / resp_fs
ts_eda = np.arange(len(eda_f)) / eda_fs
ts_temp = np.arange(len(temp_f)) / temp_fs

ecg_fn = DataNormalizer().standardize_data(ecg_f)
ppg_fn = DataNormalizer().standardize_data(ppg_f)
resp_fn = DataNormalizer().standardize_data(resp_f)
resp_fn_2 = DataNormalizer().standardize_data(resp_f_2)
eda_fn = DataNormalizer().normalize_data(eda_f)
temp_fn = DataNormalizer().standardize_data(temp_f)

#%%
########################################<<<  Bio-signal Quality Check  >>>######################################## 

#%%
                                        ############# ((ECG)) #############

quality_ecg = nk.ecg_quality(ecg_fn, sampling_rate=ecg_fs) 

# Set a threshold to filter out values above the average quality
threshold_ecg = np.mean(quality_ecg)*0.9 #TODO: adjust value 
# Print the threshold and the length of the filtered ECG signal
print(f"Quality Threshold: {threshold_ecg:.2f}")

# Replace values below the threshold with NaN
ecg_clean = np.copy(ecg_fn)
ecg_clean[quality_ecg < threshold_ecg] = np.nan
# Get the indices of the replaced values
replaced_idx_ecg = np.where(np.isnan(ecg_clean))[0]

ecg_clean = SignalProcessor.replace_nan_segments(ecg_clean, replaced_idx_ecg)

print("(:ECG quality check  is done:)")

#%%
                                        ############# ((PPG)) #############

quality_ppg = nk.ppg_quality(ppg_fn, sampling_rate=ppg_fs,method='templatematch') 

# Set a threshold to filter out values above the average quality
threshold_ppg = np.mean(quality_ppg)*0.8 #TODO: adjust value 
# Print the threshold and the length of the filtered ECG signal
print(f"Quality Threshold: {threshold_ppg:.2f}")

# Replace values below the threshold with NaN
ppg_clean = np.copy(ppg_fn)
ppg_clean[quality_ppg < threshold_ppg] = np.nan

# Get the indices of the replaced values
replaced_idx_ppg = np.where(np.isnan(ppg_clean))[0]

ppg_clean = SignalProcessor.replace_nan_segments(ppg_clean, replaced_idx_ppg)

print("(:PPG quality check  is done:)")
#%%
                                        ############# ((Resp)) #############

TRAIN_BUF = 60000
BATCH_SIZE = 512
N_TRAIN_BATCHES = int(TRAIN_BUF/BATCH_SIZE)
n_epochs = 5

params = Resp_SQA.DatasetParameters(n_window=80, train_ratio=0.8)
x = Resp_SQA.samples_unsupervised(resp_fn, params.n_window)
x_train, x_test = Resp_SQA.train_test_split(x, params.train_ratio)

train_samples = x_train.reshape(x_train.shape[0], params.n_window, 1).astype("float32")
train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_samples)
    .shuffle(TRAIN_BUF)
    .batch(BATCH_SIZE)
)

# Model training

model_vae = Resp_SQA.VAE(
    Resp_SQA.encoder_layers,
    Resp_SQA.decoder_layers,
    optimizer = tf.keras.optimizers.Adam(1e-3),
)

for epoch in range(n_epochs):
    print(f'Epoch {epoch}...')
    for batch, train_x in zip(range(N_TRAIN_BATCHES), train_dataset):
        model_vae.train_step(train_x)

test_datasets = [resp_fn]

for d in test_datasets:
    x = Resp_SQA.samples_unsupervised(d, params.n_window)
    x_hat = model_vae.reconstruct(np.atleast_3d(x)).numpy()[:, :, 0]
    
    pos = params.n_window // 2  # take the middle sample from the window
    se = np.square(x - x_hat)[:, pos]
    # Difference between real signal and reconstructed signal
    difference = x[:, pos] - x_hat[:, pos]
    
processed_vector = SignalProcessor.resp_help_function(np.abs(difference), window_size=1500,overlap_percent=0.7)

# Ensure processed_vector is the same length as resp_fn
if processed_vector.size < resp_fn.size:
    processed_vector = np.pad(processed_vector, (0, len(resp_fn) - len(processed_vector)), 'constant')
elif len(processed_vector) > len(resp_fn):
    processed_vector = processed_vector[:len(resp_fn)]

# Replace values above the threshold with a unique value (e.g., NaN)
resp_clean = np.copy(resp_fn)
resp_clean[processed_vector == 1] = np.nan

# Get the indices of the replaced values
replaced_idx_resp = np.where(processed_vector == 1)[0]

# Replace NaN values with zeroes
resp_clean = SignalProcessor.replace_nan_segments(resp_clean, replaced_idx_resp)

quality_resp = nk.ppg_quality(resp_fn, sampling_rate=resp_fs,method='templatematch') 

print("(:Resp quality check  is done:)")

# %%
                                        ############# ((EDA)) #############


# Create initial DataFrame
df_eda = pd.DataFrame({
    'EDA': eda_fn[1:],
    'Time': ts_eda[1:]
})

# Check for NaN values
print(df_eda.isna().sum())
    
EDA_SQA.compute_eda_artifacts(df_eda)

# Read CSV file
eda_data = pd.read_csv(r'artifacts-removed.csv')
quality_eda = SignalProcessor.replace_consecutive_ones(eda_data['Artifact'],consecutive_one=60)

# Replace values above the threshold with a unique value (e.g., NaN)
eda_clean = np.copy(eda_data['EDA'])
eda_clean[np.array(quality_eda) == 1] = np.nan

# Get the indices of the replaced values
replaced_idx_eda = np.where(np.array(quality_eda) == 1)[0]

# Replace NaN values with zeroes
# eda_clean[np.isnan(eda_clean)] = np.nan
eda_clean = SignalProcessor.replace_nan_segments(eda_clean, replaced_idx_eda)

ts_eda = np.arange(len(eda_clean)) / eda_fs

print("(:EDA quality check  is done:)")
#%%
########################################<<<  Bio-signal Feature Extraction  >>>######################################## 


#%%

                                    ############# ((ECG)) #############

# r-peak Detetion 
# Ensure there are no NaN values in ecg_clean before peak detection
ecg_clean_no_nan = np.copy(ecg_clean)
if np.any(np.isnan(ecg_clean_no_nan)):
    # Interpolate NaN values
    nans = np.isnan(ecg_clean_no_nan)
    x = np.arange(len(ecg_clean_no_nan))
    ecg_clean_no_nan[nans] = np.interp(x[nans], x[~nans], ecg_clean_no_nan[~nans])

signals_ecg, _ = nk.ecg_peaks(ecg_clean_no_nan, sampling_rate=ecg_fs, correct_artifacts=False, show=False)
r_peaks = [i for i,j in enumerate(signals_ecg['ECG_R_Peaks']) if j!=0]

# Convert r_peaks to numpy array for efficiency
r_peaks = np.array(r_peaks)
# Calculate IBI (Inter-Beat Interval) using heartpy
ibi_ECG = hp.analysis.calc_rr(r_peaks, sample_rate=ecg_fs)
# Create DataFrame for IBI
pd_rpeaks = pd.DataFrame({'ibi': ibi_ECG['RR_list']})
# Set index to corresponding timestamps
pd_rpeaks.index = ts_ecg[r_peaks[:-1]]
# Convert index to datetime
pd_rpeaks.index = pd.to_datetime((pd_rpeaks.index * 1000).astype(int), unit='ms', utc=True)

# Extract HRV features using flirt
hrv_features_ecg = flirt.get_hrv_features(
    pd_rpeaks['ibi'], 
    window_size, 
    window_step_size, 
    ["td", "fd", "stat", "nl"], 
    threshold = 0.5
)

print("(:ECG Feature Extraction is done:)")

#%%
                                    ############# ((PPG)) #############

# Ensure there are no NaN values in ppg_clean before peak detection
ppg_clean_no_nan = np.copy(ppg_clean)
if np.any(np.isnan(ppg_clean_no_nan)):
    # Interpolate NaN values
    nans = np.isnan(ppg_clean_no_nan)
    x = np.arange(len(ppg_clean_no_nan))
    ppg_clean_no_nan[nans] = np.interp(x[nans], x[~nans], ppg_clean_no_nan[~nans])

signals_ppg = nk.ppg_findpeaks(ppg_clean_no_nan, sampling_rate=ppg_fs, show=False)

# Calculate IBI (Inter-Beat Interval) using heartpy
ibi_PPG = hp.analysis.calc_rr(signals_ppg['PPG_Peaks'], sample_rate=ppg_fs)
# Create DataFrame for IBI
pd_PPG_peaks = pd.DataFrame({'ibi': ibi_PPG['RR_list']})
# Set index to corresponding timestamps
pd_PPG_peaks.index = ts_ppg[signals_ppg['PPG_Peaks'][:-1]]
# Convert index to datetime
pd_PPG_peaks.index = pd.to_datetime((pd_PPG_peaks.index * 1000).astype(int), unit='ms', utc=True)

# Extract HRV features using flirt
hrv_features_ppg = flirt.get_hrv_features(
    pd_PPG_peaks['ibi'], 
    window_size, 
    window_step_size, 
    ["td", "fd", "stat", "nl"], 
    threshold = 0.5
)

print("(:PPG Feature Extraction is done:)")

#%%
                                  ############# ((Resp)) #############
# Ensure there are no NaN values in resp_clean before peak detection
resp_clean_no_nan = np.copy(resp_clean)
if np.any(np.isnan(resp_clean_no_nan)):
    # Interpolate NaN values 
    nans = np.isnan(resp_clean_no_nan)
    x = np.arange(len(resp_clean_no_nan))
    resp_clean_no_nan[nans] = np.interp(x[nans], x[~nans], resp_clean_no_nan[~nans])

#Computes Respiratory Rate
resp_rate = nk.rsp_rate(resp_clean_no_nan, sampling_rate=resp_fs, method="trough")  #method = xcorr
#Computes Respiratory Volume per Time (RVT)
resp_rvt = nk.rsp_rvt(resp_clean,sampling_rate=resp_fs, method="power2020", show=False)

#amplitude of resp
peak_resp, _ = nk.rsp_peaks(resp_fn)
peak_resp_beta = [i for i,j in enumerate(peak_resp['RSP_Peaks']) if j!=0]

amp_resp = nk.rsp_amplitude(resp_fn, peak_resp)

_,resp_rate = SignalProcessor.rolling_window_sum(resp_rate, resp_fs, window_size, window_step_size) 
_,resp_rvt = SignalProcessor.rolling_window_sum(resp_rvt, resp_fs, window_size, window_step_size) 
_,amp_resp = SignalProcessor.rolling_window_sum(amp_resp, resp_fs, window_size, window_step_size) 
       
print("(:Resp Feature Extraction is done:)")

#%%
                                    ############# ((EDA)) #############

eda_ch = pd.DataFrame({'EDA': np.array(eda_clean) })
# eda_ch.index = ts_eda
eda_ch.index = pd.to_datetime((np.arange(len(eda_ch)) * 1000 / eda_fs), unit='ms', utc=True)

# Extract HRV features using flirt
eda_ch_features = flirt.eda.get_eda_features(eda_ch['EDA'], 
             window_length=window_size, 
             window_step_size=window_step_size,
             data_frequency=eda_fs)

print("(:EDA Feature Extraction is done:)")

#%%
                                  ############# ((Temp)) #############

# mean_temp = SignalProcessor.time_series_windowing_uneven(temp_fn, temp_fs, window_size, overlap_window)
time_dummy,mean_temp = SignalProcessor.rolling_window_sum(temp_fn, temp_fs, window_size, window_step_size) 
time_dummy = [ i+window_size/2 for i in time_dummy]
temp_derivative = np.diff(temp_fn) / np.diff(ts_temp[:len(temp_fn)])
_,derivative_temp = SignalProcessor.rolling_window_sum(temp_derivative, temp_fs, window_size, window_step_size) 

print("(:Temp Feature Extraction is done:)")

#%%
########################################<<<  Change Point Detection  >>>######################################## 

                                   ############# ((ECG)) #############

ECG_list = [          hrv_features_ecg['hrv_hf'],
                      hrv_features_ecg['hrv_max'],
                      hrv_features_ecg['hrv_mean_hr'],
                      hrv_features_ecg['num_ibis'],
                      hrv_features_ecg['hrv_rmssd'],
                      hrv_features_ecg['hrv_entropy'],
                      hrv_features_ecg['hrv_SD2SD1'],
                      hrv_features_ecg['hrv_lf_hf_ratio'],
                      ]
ECG_label = ['HF', 'HR MAX', 'HR Mean', 'IBIS', 'RMSSD', 'Entropy', 'SD2SD1', 'LF/HF']

                                    ############# ((PPG)) #############

PPG_list = [          hrv_features_ppg['hrv_hf'],
                      hrv_features_ppg['hrv_max'],
                      hrv_features_ppg['hrv_mean_hr'],
                      hrv_features_ppg['num_ibis'],
                      hrv_features_ppg['hrv_rmssd'],
                      hrv_features_ppg['hrv_entropy'],
                      hrv_features_ppg['hrv_SD2SD1'],
                      hrv_features_ppg['hrv_lf_hf_ratio'],
                      ]
PPG_label = ['pHF', 'pHR MAX', 'pHR Mean', 'pIBIS', 'pRMSSD', 'pEntropy', 'pSD2SD1', 'pLF/HF']

                                    ############# ((Res)) #############

Resp_list = [resp_rate, resp_rvt, amp_resp]
Resp_label = ['Resp Rate', 'Resp RVT', 'Resp Amplitude']

                                    ############# ((EDA)) #############

EDA_list = [          eda_ch_features['tonic_mean'],
                      eda_ch_features['tonic_max'],
                      eda_ch_features['tonic_energy'],
                      eda_ch_features['tonic_n_above_mean'],
                      eda_ch_features['phasic_mean'],
                      eda_ch_features['phasic_max'],
                      eda_ch_features['phasic_energy'],
                      eda_ch_features['phasic_n_above_mean'],
                      ]

EDA_label = ['Tonic Mean', 'Tonic Max', 'Tonic Energy', 'Tonic aMean',
              'Phasic Mean', 'Phasic Max', 'Phasic Energy', 'Phasic aMean']


                                    ############# ((Temp)) #############

min_length = len(mean_temp)
mean_temp = mean_temp[:min_length]
derivative_temp = derivative_temp[:min_length]

temp_PTT_list = [mean_temp,derivative_temp]
temp_PTT_label = ['Mean Temp','dMean Temp']


#%%

# Find minimum length among all signals
min_length = min(len(signal) for signal in ECG_list + PPG_list + Resp_list + EDA_list + temp_PTT_list)

# Trim all signals to minimum length
ECG_list = [signal[:min_length] for signal in ECG_list]
PPG_list = [signal[:min_length] for signal in PPG_list]  
Resp_list = [signal[:min_length] for signal in Resp_list]
EDA_list = [signal[:min_length] for signal in EDA_list]
temp_PTT_list = [signal[:min_length] for signal in temp_PTT_list]
time_dummy = time_dummy[:min_length]

# Combine all lists into a single list
all_signals = ECG_list + PPG_list + Resp_list + EDA_list + temp_PTT_list

# Combine all labels into a single list
all_labels = ECG_label + PPG_label + Resp_label + EDA_label + temp_PTT_label

# Sort time labels to handle text positioning
time_event_pairs = list(zip(time_labels, labels_text))
time_event_pairs.sort(key=lambda x: x[0])

#%%
  
# Convert list of signals to numpy array and transpose to get correct shape
all_signals_array = np.array(all_signals).T
# Normalize signals
normalized_signal = np.zeros_like(all_signals_array)
for i in range(all_signals_array.shape[1]):
    signal = all_signals_array[:, i]
    normalized_signal[:, i] = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

# Calculate percentage of missing values for each signal
missing_percentages = np.mean(np.isnan(normalized_signal), axis=0) * 100

# Get indices of signals with less than 10% missing values 
valid_signal_indices = np.where(missing_percentages < 10)[0]

# Select only signals with less than 10% missing values
# Select signals with less than 10% missing values first
signals_low_missing = normalized_signal[:, valid_signal_indices]
valid_labels = [all_labels[i] for i in valid_signal_indices]

# Calculate variance for each signal
variances = np.var(signals_low_missing, axis=0)
high_var_indices = np.where(variances > np.percentile(variances, 10))[0]

# Calculate correlation matrix
corr_matrix = np.corrcoef(signals_low_missing.T)
highly_correlated = np.zeros(len(high_var_indices), dtype=bool)

# For each signal, check if it's highly correlated with any previous signal
correlation_threshold = 0.9
for i in range(len(high_var_indices)):
    for j in range(i):
        if abs(corr_matrix[high_var_indices[i], high_var_indices[j]]) > correlation_threshold:
            highly_correlated[i] = True
            break

# Keep only signals with high variance and low correlation
final_indices = high_var_indices[~highly_correlated]

# Create final signal array
Used_signals_for_cpd = signals_low_missing[:, final_indices]

#%% 

# Separate data from video start to video end
video_end_time = labels_time[Subject_ID].loc[labels_time[Subject_ID]['Event'] == 'Video End', 'seconds_from_start'].values[0]
practice_start_time = labels_time[Subject_ID].loc[labels_time[Subject_ID]['Event'] == 'Practice Start', 'seconds_from_start'].values[0]
practice_end_time = labels_time[Subject_ID].loc[labels_time[Subject_ID]['Event'] == 'Practice End', 'seconds_from_start'].values[0]
experiment_start_time = labels_time[Subject_ID].loc[labels_time[Subject_ID]['Event'] == 'Experiment Start', 'seconds_from_start'].values[0]
timer_start_time = labels_time[Subject_ID].loc[labels_time[Subject_ID]['Event'] == 'Timer Start', 'seconds_from_start'].values[0]
experiment_end_time = labels_time[Subject_ID].loc[labels_time[Subject_ID]['Event'] == 'Clear Text', 'seconds_from_start'].values[0]

CLIP_VALUE = [int(video_start_time), int(video_end_time), int(practice_start_time), int(practice_end_time),
              int(experiment_start_time),int(timer_start_time),int(experiment_end_time)]  # Update clip value to video end

# Find the index of closest values in CLIP_VALUE in vector time_dummy
def find_closest_indices(values, vector):
    indices = []
    for value in values:
        vector_array = np.array(vector)
        index = np.argmin(np.abs(vector_array - value))
        indices.append(index)
    return indices

clip_indices = find_closest_indices(CLIP_VALUE, time_dummy)
print(f"Indices of closest values in CLIP_VALUE: {clip_indices}")

#%%
from collections import deque
from random import random

# Initialize list to store IKS statistics for all columns
all_iks_statistics = []

# Iterate over each column in Used_signals_for_cpd
for col in range(Used_signals_for_cpd.shape[1]):
    initial = Used_signals_for_cpd[clip_indices[0]:clip_indices[1]-int(checkpoint_value), col]
    stream = Used_signals_for_cpd[:, col]

    iks_statistics = []  # collect statistics generated by IKS
    iks = IKS()
    sliding = deque()  # sliding window

    for val in initial:
        iks.Add((val, random()), 0)
        wrnd = (val, random())  # we only need to keep RND component for values in the sliding window
        iks.Add(wrnd, 1)
        sliding.append(wrnd)

    # process sliding window
    for val in stream:
        iks.Remove(sliding.popleft(), 1)
        wrnd = (val, random())
        iks.Add(wrnd, 1)
        sliding.append(wrnd)

        iks_statistics.append(iks.KS())

    all_iks_statistics.append(iks_statistics)

# Calculate the average IKS statistics across all columns
average_iks_statistics = np.mean(all_iks_statistics, axis=0)

# Plot the average IKS statistics
plt.figure(figsize=(4, 4))
plt.plot(time_dummy, average_iks_statistics, label='Average IKS Statistics')

# Add vertical lines and labels for time labels in black
for time_label, event_label in time_event_pairs:
    # Draw vertical lines for ±1 second region
    if event_label != 'Video Start':
        time_label = time_label - window_size / 2  # Adjust time related to window

    if event_label == 'Timer Start':
        construction_1_start = time_label
    if event_label == 'Construction 2 Clear':
        construction_1_end = time_label
        plt.axvspan(construction_1_start, construction_1_end, color='yellow', alpha=0.3)

    plt.axvline(x=time_label, color='green', linestyle='--', alpha=0.3)  # Add vertical line
    rotation_angle = 90  # Adjust rotation angle as needed

    if event_label in ['Video Start', 'Survey 1 Start', 'Practice Start', 'Survey 2 Start',
                       'Experiment Start', 'Timer Start', 'Construction 1 Start',
                       'Survey 3 Start', 'Construction 2 Start']:
        plt.text(time_label, 0.4,  # Adjust vertical position as needed
                 event_label,
                 rotation=rotation_angle,
                 ha='right',
                 va='bottom',
                 fontsize=8,
                 color='green')
    elif event_label not in [ 'Timer Start', 'Pace Car Start', 'Timer Stop', 'Amazon Truck', 'Clear Text']:
        plt.text(time_label, 0.1,  # Adjust vertical position as needed
                 event_label,
                 rotation=rotation_angle,
                 ha='right',
                 va='bottom',
                 fontsize=8,
                 color='darkgreen')

# Add missed events text at the top
missed_events_text = "Missed Events: " + ", ".join(missed_events)
plt.suptitle(missed_events_text, color='red', fontsize=8, ha='center', y=0.95)


# Find the first two maximum points in the average_iks_statistics
max_indices = np.argpartition(average_iks_statistics, -2)[-1:]
# max_indices = max_indices[np.argsort(average_iks_statistics[max_indices])][::-1]

# Mark the first two maximum points with red circles
for idx in max_indices:
    plt.plot(time_dummy[idx], average_iks_statistics[idx], 'ro', label='Max Point')

plt.xlabel('Time')
plt.ylabel('IKS Statistic')
plt.show()
                                 

# %%

############################################################ PLOT FOR BMES CONFERENCE PAPER ############################################################
# Only plot the data after second 1000
plot_start_time = 1000
start_idx = np.searchsorted(time_dummy, plot_start_time)

# Slice all relevant arrays
plot_time = time_dummy[start_idx:]
plot_iks = average_iks_statistics[start_idx:]
plot_signals = signals_low_missing[start_idx:, :]
# For overlayed features, use the same indices

import scienceplots
import pandas as pd

with plt.style.context('science'):

    plt.figure(figsize=(6, 4))
    plt.plot(plot_time, plot_iks, label='Average IKS Statistics', linewidth=4, color='b')

    # Add red dot on the max point of the average_iks_statistics signal (after 1000s)
    if len(plot_iks) > 0:
        max_idx = np.argmax(plot_iks)
        plt.plot(plot_time[max_idx], plot_iks[max_idx], 'ro', label='Max Point')

    # Add vertical lines and labels for time labels in black, only if event label is available (not empty or NaN)
    for time_label, event_label in time_event_pairs:
        if time_label < plot_start_time:
            continue
        if event_label in [
            'Video Start', 'Practice Start', 'Timer Start',
            'Experiment Start', 'Construction 1 Start',
            'Video End', 'Practice End', 'Construction 2 End', 'Timer Stop', 'Amazon Truck','Experiment End'
        ]:
            # Draw vertical lines for ±1 second region
            if event_label != 'Video Start':
                time_label_adj = time_label - window_size / 2
            else:
                time_label_adj = time_label

            if event_label == 'Timer Start':
                construction_1_start = time_label_adj
            if event_label == 'Construction 2 Clear':
                construction_1_end = time_label_adj
                plt.axvspan(construction_1_start, construction_1_end, color='yellow', alpha=0.3)

            plt.axvline(x=time_label_adj, color='green', linestyle='--', alpha=0.3)
            rotation_angle = 90

            if event_label in [
                'Video Start', 'Practice Start', 'Timer Start',
                'Experiment Start', 'Construction 1 Start'
            ]:
                plt.text(time_label_adj, 0.5,
                        event_label,
                        rotation=rotation_angle,
                        ha='right',
                        va='bottom',
                        fontsize=10,
                        color='green')
            elif event_label in [
                'Video End', 'Practice End', 'Construction 2 End', 'Timer Stop', 'Amazon Truck','Experiment End'
            ]:
                plt.text(time_label_adj, 0.1,
                        event_label,
                        rotation=rotation_angle,
                        ha='right',
                        va='bottom',
                        fontsize=10,
                        color='darkgreen')

    def highlight_region(event_start, event_end, color='yellow', alpha=0.3):
        try:
            start_time = labels_time[Subject_ID].loc[labels_time[Subject_ID]['Event'] == event_start, 'seconds_from_start'].values[0]
            end_time = labels_time[Subject_ID].loc[labels_time[Subject_ID]['Event'] == event_end, 'seconds_from_start'].values[0]
            if start_time < plot_start_time or end_time < plot_start_time:
                return
            if event_start != 'Video Start':
                start_time = start_time - window_size / 2
                end_time = end_time - window_size / 2
            plt.axvspan(start_time, end_time, color=color, alpha=alpha)
        except Exception as e:
            print(f"Could not highlight region {event_start} to {event_end}: {e}")

    # highlight_region('Video Start', 'Video End')
    # highlight_region('Practice Start', 'Practice End')
    highlight_region('Experiment Start', 'Experiment End')

    # Overlay selected bio-signal features on the same plot, each in its own vertical zone (no overlap)
    feature_colors = ['gray']
    n_features = len(final_indices)
    zone_height = 0.8 / n_features

    for i, idx in enumerate(final_indices):
        feature = plot_signals[:, idx]
        feature_min = np.nanmin(feature)
        feature_max = np.nanmax(feature)
        if feature_max - feature_min != 0:
            feature_norm = (feature - feature_min) / (feature_max - feature_min)
        else:
            feature_norm = feature
        y_offset = 0.1 + i * zone_height
        feature_zone = y_offset + feature_norm * zone_height
        plt.plot(plot_time, feature_zone, color=feature_colors[i % len(feature_colors)], alpha=0.7)
        plt.text(plot_time[-1] + (plot_time[-1] - plot_time[0]) * 0.01, y_offset + 0.5 * zone_height, valid_labels[idx],
                color=feature_colors[i % len(feature_colors)], va='center', fontsize=8, fontweight='bold', clip_on=False)

    plt.legend(['Average KS Statistics'], loc='upper left', fontsize=7)
    plt.xlabel('Time (s)')
    plt.ylabel('KS Score')
    plt.tight_layout()
    plt.show()





# %%

############################################################ PLOT FOR BMES CONFERENCE PAPER (V2) ############################################################
# Only plot the data after second 1000
plot_start_time = 1000
start_idx = np.searchsorted(time_dummy, plot_start_time)

# Slice all relevant arrays
plot_time = time_dummy[start_idx:]
plot_iks = average_iks_statistics[start_idx:]
plot_signals = signals_low_missing[start_idx:, :]

import scienceplots
import pandas as pd

# --- User selection for representative features ---
# List all available valid_labels for user reference
print("Available features for plotting:")
for i, lbl in enumerate(valid_labels):
    print(f"{i}: {lbl}")

# Define your choices here by index (from the printed list above)
# Example: ECG = 0, PPG = 1, EDA = 2, Resp = 3 (update as needed)
# You can change these indices to select different features
selected_indices = {
    'ECG': 0,   # <-- change to your preferred ECG feature index
    'PPG': 11,   # <-- change to your preferred PPG feature index
    'EDA': 19,   # <-- change to your preferred EDA feature index
    'Resp': 17   # <-- change to your preferred Resp feature index
}
# If you want to skip a modality, set its value to None

feature_colors = ['tab:red', 'tab:green', 'tab:purple', 'tab:orange']
modality_names = list(selected_indices.keys())
selected_idxs = [selected_indices[m] for m in modality_names]
selected_labels = [valid_labels[idx] if idx is not None else 'N/A' for idx in selected_idxs]

with plt.style.context('science'):
    fig, (ax_ks, ax_feat) = plt.subplots(2, 1, figsize=(5, 5), sharex=True, 
                                         gridspec_kw={'height_ratios': [1, 2]})

    # --- Top: KS Score ---
    ax_ks.plot(plot_time, plot_iks, label='Average IKS Statistics', linewidth=3, color='b')
    if len(plot_iks) > 0:
        max_idx = np.argmax(plot_iks)
        ax_ks.plot(plot_time[max_idx], plot_iks[max_idx], 'ro', label='Max Point')
    ax_ks.set_ylabel('KS Score')
    ax_ks.legend(fontsize=8, loc='upper left')
    # ax_ks.set_title('KS Score (Change Point Detection)')

    # --- Bottom: Representative Features ---
    y_offset = 0.1
    n_features = sum(idx is not None for idx in selected_idxs)
    zone_height = 0.8 / max(n_features, 1)
    for i, (idx, label, color, modality) in enumerate(zip(selected_idxs, selected_labels, feature_colors, modality_names)):
        if idx is None:
            continue
        feature = plot_signals[:, idx]
        feature_min = np.nanmin(feature)
        feature_max = np.nanmax(feature)
        if feature_max - feature_min != 0:
            feature_norm = (feature - feature_min) / (feature_max - feature_min)
        else:
            feature_norm = feature
        feature_zone = y_offset + feature_norm * zone_height
        ax_feat.plot(plot_time, feature_zone, color=color, alpha=0.8, label=f'{modality}: {label}')
        y_offset += zone_height
    ax_feat.set_ylabel('Features (normalized)')
    ax_feat.set_yticks([]) 
    ax_feat.set_xlabel('Time (s)')
    ax_feat.legend(fontsize=8, loc='upper left')

    # --- Add event lines and labels to both axes ---
    for ax in [ax_ks, ax_feat]:
        for time_label, event_label in time_event_pairs:
            if time_label < plot_start_time:
                continue
            # Replace 'Amazon Truck' with 'Delivery Truck' in event labels
            display_label = event_label.replace('Amazon Truck', 'Delivery Truck')
            if event_label in [
                'Video Start', 'Practice Start', 'Timer Start',
                'Experiment Start', 'Construction 1 Start',
                'Video End', 'Practice End', 'Construction 2 End', 'Timer Stop', 'Amazon Truck','Experiment End'
            ]:
                if event_label != 'Video Start':
                    time_label_adj = time_label - window_size / 2
                else:
                    time_label_adj = time_label
                ax.axvline(x=time_label_adj, color='green', linestyle='--', alpha=0.3)
                
                if ax is ax_ks:
                    rotation_angle = 90
                    if event_label in [
                        'Video Start', 'Practice Start', 
                       
                    ]:
                        ax.text(time_label_adj, ax.get_ylim()[1]*0.85,
                                display_label, rotation=rotation_angle,
                                ha='right', va='bottom', fontsize=10, color='green')
                        
                    elif event_label in [
                        'Video End', 'Practice End', 'Construction 1 Start', 'Timer Stop', 'Amazon Truck','Experiment End','Experiment Start'
                    ]:
                        
                        ax.text(time_label_adj, ax.get_ylim()[0]+0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                                display_label, rotation=rotation_angle,
                                ha='right', va='bottom', fontsize=10, color='darkgreen')

    plt.tight_layout()
    plt.show()
    # Save the current figure to file
    fig.savefig(os.path.join(os.path.dirname(__file__), f'KS_and_features_{Subject_ID}_after1000s.png'), dpi=300, bbox_inches='tight')
    print(f"Saved figure to {os.path.join(os.path.dirname(__file__), f'KS_and_features_{Subject_ID}_after1000s.png')}")
#%%
# Save KS signal and event locations to a CSV file (only after 1000 sec)

# Find event locations (indices in plot_time) for key events
def find_event_index(event_name):
    try:
        event_time = labels_time[Subject_ID].loc[labels_time[Subject_ID]['Event'] == event_name, 'seconds_from_start'].values[0]
        if event_name != 'Video Start':
            event_time = event_time - window_size / 2
        idx = np.searchsorted(plot_time, event_time)
        return idx
    except Exception:
        return None

event_names = ['Experiment Start', 'Experiment End', 'Amazon Truck', 'Construction 1 Start', 'Construction 2 End']
event_indices = {name: find_event_index(name) for name in event_names}

# Prepare data to save (only after 1000 sec)
save_dict = {
    'time': plot_time,
    'ks_signal': plot_iks
}
for name, idx in event_indices.items():
    save_dict[f'{name}_idx'] = [idx] * len(plot_time)  # Save as column for easy CSV writing

# Save as DataFrame
df_save = pd.DataFrame(save_dict)
output_file = os.path.join(os.path.dirname(__file__), f'KS_and_events_{Subject_ID}_after1000s.csv')
df_save.to_csv(output_file, index=False)
print(f"Saved KS signal and event indices (after 1000s) to {output_file}")

# %%




# %%
