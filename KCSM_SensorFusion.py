
#%% 
########################################<<<  Load Libraries  >>>######################################## 

from Load.LoadData import DataProcessor, SkinTempProcessor, DataNormalizer 
from Utils.func import SignalProcessor 
from PreProcessing import PPG_SQA, Resp_SQA, EDA_SQA
import os
import matplotlib.pyplot as plt
from scipy import signal
import neurokit2 as nk
import numpy as np 
import tensorflow as tf
import pandas as pd
import heartpy as hp
import flirt
from CPD.CPD import CPD_model, CPD_Plotting
from DTW.dtw import accelerated_dtw
import seaborn as sns
import matplotlib.colors as colors
from datetime import datetime
from CPD.KCPD import KernelCPD

%matplotlib widget
%load_ext autoreload
%autoreload 2
# %%
########################################<<<  Load Data  >>>######################################## 

Subject_ID = 'P14S' #'P10S'
# Define clipping value for the beginning and end of the signal

# Define the file path and lines to skip
file_path = os.path.join(r'c:\Users\573751\OneDrive - TMNA\Desktop\Files\Datasets\Neq15\Surprise', Subject_ID + '.txt')
lines_to_skip = 20  # Adjust this number based on how many lines you want to skip

biodata = DataProcessor(file_path).process_file(lines_to_skip)

# Print the results
for i, length, data in biodata:
    print(f"Column {i+1}: {length} entries")
    print(f"Sample data: {data[:5]}...")  # Print first 5 entries as a sample
    print("(:Data Loading is done:)")  


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

input_file =  r"c:\Users\573751\OneDrive - TMNA\Desktop\Files\Datasets\Neq15\Sensor Fusion 2024\Time_Stamps\Experiment_2_Stamps_S.xlsx"
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

#if end of study is missing set it -1
if study_end_time < 0:
    study_end_time = -1

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

# Plot the standardized ECG signal
plt.figure(figsize=(10, 4))
plt.plot(ts_ecg, ecg_fn, color='black', linewidth=2)  # Plot in black and make the line thicker
plt.axis('off')  # Remove edges and ticks
plt.show()



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

# Replace NaN values with zeroes
ecg_clean[np.isnan(ecg_clean)] = 0

# Print the indices of the replaced values
print(f"Indices of replaced values: {replaced_idx_ecg} Size of replace is: {len(replaced_idx_ecg)}")

# Plot the filtered signal with markers at the replaced indices
SignalProcessor.plot_signal_with_replaced_values(ecg_clean, replaced_idx_ecg)


#%%
                                        ############# ((PPG2)) #############

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

# Replace NaN values with zeroes
ppg_clean[np.isnan(ppg_clean)] = 0

# Print the indices of the replaced values
print(f"Indices of replaced values: {replaced_idx_ppg} Size of replace is: {len(replaced_idx_ppg)}")

# Plot the filtered signal with markers at the replaced indices
SignalProcessor.plot_signal_with_replaced_values(ppg_clean, replaced_idx_ppg)

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
resp_clean[np.isnan(resp_clean)] = 0

# Print the indices of the replaced values
print(f"Indices of replaced values: {replaced_idx_resp} Size of replace is: {len(replaced_idx_resp)}")

# Plot the filtered  signal with markers at the replaced indices
SignalProcessor.plot_signal_with_replaced_values(resp_clean,replaced_idx_resp)

#Alternative:
quality_resp = nk.ppg_quality(resp_fn, sampling_rate=resp_fs,method='templatematch') 

# %%
                                        ############# ((EDA)) #############


# Create initial DataFrame
df_eda = pd.DataFrame({
    'EDA': eda_fn,
    'Time': ts_eda
})
    
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
eda_clean[np.isnan(eda_clean)] = 0

#TODO: Update time (EDA bacome shorter after the removing operation will investigate this later)
ts_eda = np.arange(len(eda_clean)) / eda_fs

# Print the indices of the replaced values
print(f"Indices of replaced values: {replaced_idx_eda} Size of replace is: {len(replaced_idx_eda)}")

# Plot the filtered signal with markers at the replaced indices
SignalProcessor.plot_signal_with_replaced_values(eda_clean, replaced_idx_eda)

#%%
########################################<<<  Bio-signal Feature Extraction  >>>######################################## 


#%%

                                    ############# ((ECG)) #############

# r-peak Detetion 
signals_ecg, _ = nk.ecg_peaks(ecg_clean, sampling_rate=ecg_fs, correct_artifacts=True, show=False)
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

#%%
                                    ############# ((PPG)) #############

signals_ppg = nk.ppg_findpeaks(ppg_clean, sampling_rate=ppg_fs, show=False)

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

#%%
                                  ############# ((Resp)) #############

#Computes Respiratory Rate
resp_rate = nk.rsp_rate(resp_clean, sampling_rate=resp_fs, method="trough")  #method = xcorr
#Computes Respiratory Volume per Time (RVT)
resp_rvt = nk.rsp_rvt(resp_clean,sampling_rate=resp_fs, method="power2020", show=False)
#amplitude of resp
peak_resp, _ = nk.rsp_peaks(resp_fn)
amp_resp = nk.rsp_amplitude(resp_fn, peak_resp)

_,resp_rate = SignalProcessor.rolling_window_sum(resp_rate, resp_fs, window_size, window_step_size) 
_,resp_rvt = SignalProcessor.rolling_window_sum(resp_rvt, resp_fs, window_size, window_step_size) 
_,amp_resp = SignalProcessor.rolling_window_sum(amp_resp, resp_fs, window_size, window_step_size) 
       

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

#%%
                                  ############# ((Temp)) #############

# mean_temp = SignalProcessor.time_series_windowing_uneven(temp_fn, temp_fs, window_size, overlap_window)
_,mean_temp = SignalProcessor.rolling_window_sum(temp_fn, temp_fs, window_size, window_step_size) 
temp_derivative = np.diff(temp_fn) / np.diff(ts_temp[:len(temp_fn)])
_,derivative_temp = SignalProcessor.rolling_window_sum(temp_derivative, temp_fs, window_size, window_step_size) 

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


#%%

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


#%%

                                    ############# ((Res)) #############

Resp_list = [resp_rate, resp_rvt, amp_resp]

Resp_label = ['Resp Rate', 'Resp RVT', 'Resp Amplitude']


#%%

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



#%%
                                    ############# ((Temp and PTT)) #############


# mean_temp = SignalProcessor.time_series_windowing_uneven(temp_fn, temp_fs, window_size, overlap_window)
time_dummy,mean_temp = SignalProcessor.rolling_window_sum(temp_fn, temp_fs, window_size, window_step_size) 
time_dummy = [ i+window_size/2 for i in time_dummy]
temp_derivative = np.diff(temp_fn) / np.diff(ts_temp[:len(temp_fn)])
_,derivative_temp = SignalProcessor.rolling_window_sum(temp_derivative, temp_fs, window_size, window_step_size) 
temp_PTT_list = [mean_temp,derivative_temp,]
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
########################################<<<  Apply Kernel Change Detection in Features  >>>######################################## 

algo_c = KernelCPD(kernel="rbf", min_size=10).fit(Used_signals_for_cpd) 
result = algo_c.predict(pen=10)[:-1] # 

bps = [0] + result + [len(normalized_signal)]


                                                 ####################### PLOT FEATURES #######################
# Create figure and axis
fig, ax1 = plt.subplots(figsize=(8, 4))

# Apply moving average to each signal with window size determined
ma_window = 1
smoothed_signals = np.zeros_like(normalized_signal)
for i in range(normalized_signal.shape[1]):
    smoothed_signals[:,i] = np.convolve(normalized_signal[:,i], 
                                       np.ones(ma_window)/ma_window, 
                                       mode='same')

# Plot each signal on its own line, vertically stacked
num_signals = normalized_signal.shape[1]
spacing = 1.0 / num_signals

for i in range(normalized_signal.shape[1]):
    offset = 1.0 - (i * spacing)
    ax1.plot(time_dummy, smoothed_signals[:,i] * spacing + offset - spacing/2, 
             label=all_labels[i], linewidth=1,color='black',alpha=0.8)
    # Add label on right side
    ax1.text(time_dummy[-1] + 100, offset - spacing/2, all_labels[i], 
                 verticalalignment='center', fontsize=8)
    

                                                 ####################### VERTICAL LINES AND ARROWS FOR CHANGES #######################
                                                   
# Add vertical lines for detected change points in red

# Filter change points to keep only those within the experiment start and end times
experiment_start_time = labels_time[Subject_ID].loc[labels_time[Subject_ID]['Event'] == 'Experiment Start', 'seconds_from_start'].values[0]
experiment_end_time = labels_time[Subject_ID].loc[labels_time[Subject_ID]['Event'] == 'Clear Text', 'seconds_from_start'].values[0]

filtered_CP = [cp for cp in result if experiment_start_time <= time_dummy[cp] < experiment_end_time]

for change_point in filtered_CP:
    ax1.axvline(x=time_dummy[change_point], color='red', linestyle='--', alpha=0.5)
    

                                                ####################### EVENTS LABELS #######################

# Add vertical lines and labels for time labels in black
for time_label, event_label in time_event_pairs:
    # Draw vertical lines for ±1 second region
    # if event_label != 'Video Start':
    time_label = time_label - video_start_time #- window_size/2  #TODO to fix time related to window
    if event_label in ['Crash','Barrel']: ax1.axvspan(time_label - 5, time_label + window_size/2, color='green', alpha=0.1)  # Add shaded region ±1 second
    ax1.axvline(x=time_label, color='green', linestyle='--', alpha=0.3)  # Add vertical line
    rotation_angle = 90  # Adjust rotation angle as needed
    if event_label in ['Video Start', 'Survey 1 Start', 'Practice Start', 'Survey 2 Start',
       'Experiment Start','Crash','Barrel',
        'Survey 3 Start', ]:
        ax1.text(time_label, 0.7,  # Adjust vertical position as needed
                event_label,
                rotation=rotation_angle,
                ha='right',
                va='bottom',
                fontsize=8,
                color='green')
    else: 
        ax1.text(time_label, 0,  # Adjust vertical position as needed
                event_label,
                rotation=rotation_angle,
                ha='right',
                va='bottom',
                fontsize=8,
                color='darkgreen')

# Add missed events text at the top
missed_events_text = "Missed Events: " + ", ".join(missed_events)
fig.suptitle(missed_events_text, color='red', fontsize=8, ha='center', y=0.95)

plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
ax1.set_yticks([])  # Remove y-axis ticks
ax1.set_ylabel('')  # Remove y-axis label

print([time_dummy[i] for i in result])

    
# %%
