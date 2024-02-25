import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks

# Function to calculate speed, velocity, and acceleration
def calculate_metrics(data, peaks):
    speeds, velocities, accelerations = [0], [0], [0]  # Initialize with starting values
    
    for i in range(1, len(peaks)):
        distance_diff = data['Distance'].iloc[peaks[i]] - data['Distance'].iloc[peaks[i-1]]
        time_diff = data['Timestamp'].iloc[peaks[i]] - data['Timestamp'].iloc[peaks[i-1]]
        if time_diff != 0:
            speed = distance_diff / time_diff
            velocity = speed  # Assuming straight line motion, speed = velocity magnitude
            acceleration = (velocity - velocities[-1]) / time_diff if i > 1 else 0
        else:
            speed, velocity, acceleration = 0, 0, 0
        
        speeds.append(speed)
        velocities.append(velocity)
        accelerations.append(acceleration)
    
    return speeds, velocities, accelerations

# Function to plot hand movement and calculate metrics
def plot_hand_movement(data, max_width, min_width, max_threshold_multiplier, min_threshold_multiplier):
    y = data['Distance'].values
    timestamps = data['Timestamp'].values
    median_val = np.median(y)
    
    # Define thresholds
    max_threshold = median_val * max_threshold_multiplier
    min_threshold = median_val * min_threshold_multiplier
    
    # Find peaks
    max_peaks, _ = find_peaks(-y, width=max_width)
    min_peaks, _ = find_peaks(y, width=min_width)
    
    # Filter peaks based on thresholds
    filtered_max_peaks = max_peaks[y[max_peaks] < max_threshold]
    filtered_min_peaks = min_peaks[y[min_peaks] > min_threshold]
    
    all_peaks = np.sort(np.concatenate((filtered_max_peaks, filtered_min_peaks)))
    
    # Calculate metrics
    speeds, velocities, accelerations = calculate_metrics(data, all_peaks)
    
    peaks_df = pd.DataFrame({
        'Timestamp': data['Timestamp'][all_peaks],
        'Distance': data['Distance'][all_peaks],
        'Average Speed': speeds,
        'Velocity': velocities,
        'Acceleration': accelerations
    })
    
    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=y, mode='lines', name='Hand Movement'))
    fig.add_trace(go.Scatter(x=timestamps[filtered_max_peaks], y=y[filtered_max_peaks], mode='markers', marker=dict(color='blue'), name='Max Peaks'))
    fig.add_trace(go.Scatter(x=timestamps[filtered_min_peaks], y=y[filtered_min_peaks], mode='markers', marker=dict(color='red'), name='Min Peaks'))
    fig.add_hline(y=max_threshold, line=dict(color='green', dash='dash'), name='Max Threshold')
    fig.add_hline(y=min_threshold, line=dict(color='yellow', dash='dash'), name='Min Threshold')
    
    fig.update_layout(title='Hand Movement with Dynamic Thresholds', xaxis_title='Timestamp', yaxis_title='Distance', legend=dict(y=1, x=1))
    st.plotly_chart(fig, use_container_width=True)
    
    return peaks_df

# Streamlit app interface
st.title('Hand Movement Visualization App')

uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type="csv")

for uploaded_file in uploaded_files:
    st.write(f"Processing file: {uploaded_file.name}")

    df = pd.read_csv(uploaded_file)
    df = df.iloc[:, :5]
    df.columns = ['X', 'Y', 'Z', 'T', 'Frame_num']
    df['Timestamp'] = df['Frame_num'] / 60  # Assuming 60 FPS
    df['Distance'] = np.sqrt((df['X'] - df['X'][0])**2 + (df['Y'] - df['Y'][0])**2 + (df['Z'] - df['Z'][0])**2)
    df2 = df[['Timestamp', 'Distance']]

    # User inputs for peak detection parameters
    max_width = st.slider('Select Maximum Peak Width', 1, 200, 30, key=f"max_width_{uploaded_file.name}")
    min_width = st.slider('Select Minimum Peak Width', 1, 200, 40, key=f"min_width_{uploaded_file.name}")
    max_threshold_multiplier = st.slider('Maximum Peak Threshold Multiplier', 0.5, 2.0, 1.0, 0.1, key=f"max_threshold_{uploaded_file.name}")
    min_threshold_multiplier = st.slider('Minimum Peak Threshold Multiplier', 1.0, 3.0, 1.5, 0.1, key=f"min_threshold_{uploaded_file.name}")

    peaks_df = plot_hand_movement(df2, max_width, min_width, max_threshold_multiplier, min_threshold_multiplier)

    # Download button for peaks data
    st.download_button(
        "Download Peaks Data as CSV",
        data=peaks_df.to_csv(index=False).encode('utf-8'),
        file_name=f'{uploaded_file.name.split(".")[0]}_peaks_data.csv',
        mime='text/csv',
    )
