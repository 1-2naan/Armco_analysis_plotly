import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def calculate_metrics(data, peaks):
    speeds = [0]  # Starting with an initial average speed of 0
    velocities = [0]  # Starting with 0 velocity
    accelerations = [0]  # Starting with 0 acceleration
    
    for i in range(1, len(peaks)):
        # Average Speed calculation
        distance_diff = data['Distance'].iloc[peaks[i]] - data['Distance'].iloc[peaks[i-1]]
        time_diff = data['Timestamp'].iloc[peaks[i]] - data['Timestamp'].iloc[peaks[i-1]]
        average_speed = distance_diff / time_diff if time_diff else 0
        speeds.append(average_speed)
        
        # Instantaneous Velocity calculation
        velocity = (data['Distance'].iloc[peaks[i]] - data['Distance'].iloc[peaks[i-1]]) / time_diff if time_diff else 0
        velocities.append(velocity)
        
        # Acceleration calculation
        acceleration = (velocities[i] - velocities[i-1]) / time_diff if i > 1 and time_diff else 0
        accelerations.append(acceleration)
        
    return speeds, velocities, accelerations

# Update the plot_hand_movement function to include average speed along with velocity and acceleration
def plot_hand_movement(data, max_width, min_width):
    y = data['Distance'].values
    timestamps = data['Timestamp'].values
    median_val = np.median(y)
    
    max_peaks, _ = find_peaks(-y, width=max_width)
    min_peaks, _ = find_peaks(y, width=min_width)
    
    # Filter peaks using the median value
    filtered_max_peaks = max_peaks[y[max_peaks] < median_val]
    filtered_min_peaks = min_peaks[y[min_peaks] > median_val * 1.5]  # Adjust this multiplier as needed

    # Combine and sort all peaks
    all_peaks = np.sort(np.concatenate((filtered_max_peaks, filtered_min_peaks)))

    # Calculate average speed, velocities, and accelerations
    speeds, velocities, accelerations = calculate_metrics(data, all_peaks)

    # Create a DataFrame to download
    peaks_df = pd.DataFrame({
        'Timestamp': data['Timestamp'][all_peaks],
        'Distance': data['Distance'][all_peaks],
        'Average Speed': speeds,
        'Velocity': velocities,
        'Acceleration': accelerations
    })

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, y, label='Hand Movement')
    plt.plot(timestamps[filtered_max_peaks], y[filtered_max_peaks], "bo", label='Maximum Peaks')
    plt.plot(timestamps[filtered_min_peaks], y[filtered_min_peaks], "ro", label='Minimum Peaks')
    plt.xlabel('Timestamp')
    plt.ylabel('Distance Moved')
    plt.title('Hand Movement with Velocity and Acceleration at Peaks')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)

    return peaks_df

# Streamlit app starts here
st.title('Hand Movement Visualization App')

# Allow multiple files to be uploaded
uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type="csv")

for uploaded_file in uploaded_files:
    # Show the filename
    st.write(f"Processing file: {uploaded_file.name}")

    # Read the data from the uploaded file
    df = pd.read_csv(uploaded_file)
    df.columns=['X','Y','Z','T','Frame_num']
    df[['Timestamp']]=df[['Frame_num']]/60
    df['X1']=df['X']-df['X'][0]
    df['Y1']=df['Y']-df['Y'][0]
    df['Z1']=df['Z']-df['Z'][0]
    df['Distance'] = (df['X1']**2+df['Y1']**2+df['Z1']**2)**0.5
    df2=df[['Timestamp','Distance']]

    # Sliders to adjust width for peak finding for each file
    st.write("Adjust the peak width settings for this file:")
    max_width = st.slider(f'Select Maximum Peak Width for {uploaded_file.name}', min_value=1, max_value=200, value=30, key=f"max_width_{uploaded_file.name}")
    min_width = st.slider(f'Select Minimum Peak Width for {uploaded_file.name}', min_value=1, max_value=200, value=40, key=f"min_width_{uploaded_file.name}")

    # Process the file and plot the data
    peaks_df = plot_hand_movement(df2, max_width, min_width)

    # Provide a download link for the peaks DataFrame
    st.download_button(
        label="Download Peaks Data as CSV",
        data=peaks_df.to_csv(index=False).encode('utf-8'),
        file_name=f'{uploaded_file.name.split(".")[0]}_peaks_data.csv',
        mime='text/csv',
    )