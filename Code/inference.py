#!/usr/bin/env python3

"""Real-time temporal transformer inference demo.

Open-source safe defaults are used for runtime configuration.
Set environment variables when needed:
- SERIAL_PORT (default: COM3)
- SERIAL_BAUD_RATE (default: 1000000)
- MODEL_FILE (default: temporal_transformer_model.pt)
"""

import serial
import numpy as np
import torch
import torch.nn as nn
import math
import cv2
import time
import statistics
import pyttsx3
import os
import threading
import winsound
from collections import deque

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
COM_PORT = os.getenv('SERIAL_PORT', 'COM3')
BAUD_RATE = int(os.getenv('SERIAL_BAUD_RATE', '1000000'))
MODEL_FILE = os.getenv('MODEL_FILE', 'temporal_transformer_model.pt')
RAW_DATA_POINTS = 256
DATA_POINTS = 248
IMAGE_DIR = os.path.join(SCRIPT_DIR, 'region_classes')  # Directory containing images for each class
AUDIO_DIR = os.path.join(SCRIPT_DIR, 'audio_cues')  # Directory containing audio files
CALIBRATION_DURATION = 1  # Duration for calibration in seconds
SEQUENCE_LENGTH = 7  # Sequence length for temporal transformer
SENSOR_LENGTH = 124  # Number of sensors (should match model training)

# Calibration Method Toggles
ENABLE_BEGINNING_CALIBRATION = True  # Toggle for dedicated calibration phase
ENABLE_MEAN_CALIBRATION = True  # Toggle for mean calibration method
ENABLE_ONE_CHANNEL_OUT_CALIBRATION = False  # Toggle for one channel out calibration

# Create audio directory if it doesn't exist
os.makedirs(AUDIO_DIR, exist_ok=True)

# Preload audio file paths for each class
audio_files = {}

def play_audio_cue(class_name):
    """Play audio cue asynchronously using winsound"""
    audio_file = audio_files.get(class_name)
    if audio_file and os.path.exists(audio_file):
        # Play sound asynchronously (non-blocking)
        threading.Thread(target=lambda: winsound.PlaySound(audio_file, winsound.SND_FILENAME | winsound.SND_ASYNC), daemon=True).start()
    else:
        print(f"Warning: Audio file not found for class '{class_name}'")

# ================================================================
# Model Architecture Classes ( from temporal_transformer.py)
# ================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

class SpatialFeatureExtractor(nn.Module):
    def __init__(self, n_channels: int = 2, hidden: int = 64, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_channels, hidden, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x):
        x = self.dropout1(self.act1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.act2(self.bn2(self.conv2(x))))
        
        x_avg = self.pool_avg(x)
        x_max = self.pool_max(x)
        
        x = torch.cat([x_avg, x_max], dim=1)
        x = x.squeeze(-1)
        
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, 
                 n_channels: int = 2,
                 sequence_length: int = 7,
                 sensor_length: int = 124,
                 hidden: int = 64,
                 n_classes: int = 5,
                 nhead: int = 8,
                 num_encoder_layers: int = 2,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.sensor_length = sensor_length
        self.hidden = hidden
        
        self.spatial_extractor = SpatialFeatureExtractor(
            n_channels=n_channels, 
            hidden=hidden, 
            dropout=dropout
        )
        
        self.d_model = hidden * 2
        self.pos_encoder = PositionalEncoding(self.d_model, sequence_length)
        
        # Add LayerNorm before transformer for per-batch normalization
        self.pre_transformer_norm = nn.LayerNorm(self.d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, n_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        spatial_features = []
        for t in range(self.sequence_length):
            x_t = x[:, t, :, :]
            features_t = self.spatial_extractor(x_t)
            spatial_features.append(features_t)
        
        x = torch.stack(spatial_features, dim=0)
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Apply LayerNorm for per-batch normalization
        # This normalizes across the feature dimension for each position in the batch
        x = self.pre_transformer_norm(x)
        
        # Apply transformer encoder
        # Note: Transformer expects (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x[-1, :, :]
        logits = self.classifier(x)
        
        return logits

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(SCRIPT_DIR, MODEL_FILE)
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file not found: {model_path}. "
        "Set MODEL_FILE env var to the correct model filename."
    )
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# Calculate model size
model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
model_size_mb = model_size_bytes / (1024 * 1024)
print(f"Model Size: {model_size_mb:.2f} MB ({model_size_bytes:,} bytes)")

# Initialize latency tracking
inference_latencies = []

# Initialize serial connection
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)

# Define class names based on model training (typically 5 classes for gaze regions)
CLASS_NAMES = ['Bottom Left', 'Bottom Right', 'Center', 'Top Left', 'Top Right']
N_CLASSES = len(CLASS_NAMES)

# Load audio files for each class
for cls in CLASS_NAMES:
    audio_path = os.path.join(AUDIO_DIR, f"{cls}.wav")
    audio_files[cls] = audio_path
    if not os.path.exists(audio_path):
        print(f"Warning: Audio file missing: {audio_path}")

print(f"Loaded audio files: {list(audio_files.keys())}")

# Preload images for each class into a list
class_images = []
for idx in range(N_CLASSES):
    image_path = os.path.join(IMAGE_DIR, f"class_{CLASS_NAMES[idx]}.png")
    img = cv2.imread(image_path)
    if img is None:
        # Create a readable placeholder image to avoid imshow assertion errors
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, f"{CLASS_NAMES[idx]}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        img = placeholder
        print(f"Warning: Could not read image {image_path}. Using placeholder for class '{CLASS_NAMES[idx]}'.")
    class_images.append(img)

# Create a window for displaying images
cv2.namedWindow('Predicted Class', cv2.WINDOW_NORMAL)

# # Get screen resolution to maximize the window
# screen_width, screen_height = cv2.getWindowImageRect('Predicted Class')[2:]
# if platform.system() == "Windows":
#     screen_width, screen_height = cv2.getWindowImageRect('Predicted Class')[2:]
# elif platform.system() == "Linux":
#     screen_width, screen_height = cv2.getWindowImageRect('Predicted Class')[2:]
# else:
#     screen_width, screen_height = cv2.getWindowImageRect('Predicted Class')[2:]

# Set window size to match screen resolution (maximize)
cv2.resizeWindow('Predicted Class', 640, 480)

try:
    # Initialize calibration vector
    calibration_vector = None
    
    # Calibration Phase (conditional based on toggle)
    if ENABLE_BEGINNING_CALIBRATION:
        print("Starting calibration...")

        # Audio cue
        engine = None
        try:
            engine = pyttsx3.init() # Initialize the text-to-speech engine
            engine.setProperty('rate', 150)  # Speed of speech
            engine.setProperty('volume', 1.0)  # Volume level

            # Say the starting phrase
            phrase = "Starting Calibration! Look in the top right corner."
            print(f"Speaking: {phrase}")
            engine.say(phrase)
            engine.runAndWait()
            time.sleep(1)  # Pause for 1 second before calibration

            # Start the event loop for continuous speech
            engine.startLoop(False)
        except Exception as e:
            print(f"Error initializing TTS engine: {e}")
            engine = None

        calibration_data = []
        start_time = time.time()
        while time.time() - start_time < CALIBRATION_DURATION:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                try:
                    data = np.array([float(x) for x in line.split(',')])
                    # Remove the same indices as in inference phase
                    del_col_index = [0, 32, 64, 96, 128, 160, 192, 224]
                    data = np.delete(data, del_col_index)  # Reduce from 256 to 248                    
                    if len(data) == DATA_POINTS:
                        calibration_data.append(data)
                except ValueError as e:
                    print(f"Invalid data format during calibration: {e}")
                    continue

        if calibration_data:
            phrase = "Start testing."
            print(f"Speaking: {phrase}")

            if engine is not None:
                try:             
                    while engine.isBusy():
                        engine.iterate()
                        time.sleep(0.1)

                    engine.say(phrase)
                        
                    # Wait for the final phrase to complete before ending the loop
                    while engine.isBusy():
                        engine.iterate()
                        time.sleep(2.5)

                except Exception as e:
                    print(f"Error speaking phrase '{phrase}': {e}")
                finally:
                    # Properly cleanup the TTS engine
                    try:
                        if hasattr(engine, 'endLoop'):
                            engine.endLoop()
                        if hasattr(engine, 'stop'):
                            engine.stop()
                        # Clear the reference to help with garbage collection
                        del engine
                    except Exception as cleanup_error:
                        print(f"Warning: TTS cleanup error (can be ignored): {cleanup_error}")
                    engine = None

            calibration_vector = np.mean(calibration_data, axis=0)
            print("Calibration complete. Averaging calibration values...")
        else:
            print("No calibration data collected. Proceeding without calibration.")
            calibration_vector = np.zeros(DATA_POINTS)
            # Clean up engine if it was initialized but no data was collected
            if engine is not None:
                try:
                    if hasattr(engine, 'endLoop'):
                        engine.endLoop()
                    if hasattr(engine, 'stop'):
                        engine.stop()
                    del engine
                except Exception as cleanup_error:
                    print(f"Warning: TTS cleanup error (can be ignored): {cleanup_error}")
                engine = None
    else:
        print("Beginning calibration disabled. Proceeding without dedicated calibration phase.")
        calibration_vector = np.zeros(DATA_POINTS)

    # Inference Phase
    print("Starting inference phase...")
    last_predicted_class_index = None
    last_class_change_time = time.time()
    stable_prediction_duration = 1.5  # Minimum seconds before allowing class change
    buffer = []
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)  # Buffer to store sequence of data
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            try:
                data = np.array([float(x) for x in line.split(',')]) # data.shape --> (256,)
                
                # Validate data length before processing
                if len(data) != RAW_DATA_POINTS:
                    print(f"Warning: Expected {RAW_DATA_POINTS} data points, got {len(data)}. Skipping this sample.")
                    continue
                
                del_col_index = [0, 32, 64, 96, 128, 160, 192, 224]        
                # Remove specified indices to reduce from 256 to 248 data points
                data = np.delete(data, del_col_index)    # data.shape --> (248,)     
                
                if len(data) == DATA_POINTS:
                    # Preprocess the data (apply calibration if available)
                    if calibration_vector is not None:
                        features_zeroed = data - calibration_vector
                    else:
                        features_zeroed = data
                    

                    # LEFT and RIGHT ready for inference
                    left_data = features_zeroed[::2]  # Odd indices (left sensors)
                    right_data = features_zeroed[1::2]  # Even indices (right sensors)

                    # MEAN CALIBRATION METHOD
                    # Compute the mean intensity value across all features_zeroed data, and take the mean as calibration_vector for mean calibration method.
                    # Subtract it afterwards to provide an updated left_data and right_data
                    if ENABLE_MEAN_CALIBRATION:
                        mean_calibration_left = np.mean(left_data)
                        left_data = left_data - mean_calibration_left
                        mean_calibration_right = np.mean(right_data)
                        right_data = right_data - mean_calibration_right                        
                        # print(f"Applied mean mean_calibration_left: {mean_calibration_left:.4f}")


                    # Ensure we have the right sensor length
                    if len(left_data) >= SENSOR_LENGTH and len(right_data) >= SENSOR_LENGTH:
                        left_data = left_data[:SENSOR_LENGTH]
                        right_data = right_data[:SENSOR_LENGTH]
                        
                        # Stack channels: shape (2, SENSOR_LENGTH)
                        current_sample = np.stack([left_data, right_data], axis=0)
                        sequence_buffer.append(current_sample)
                        
                        # Only run inference when we have enough samples in sequence
                        if len(sequence_buffer) == SEQUENCE_LENGTH:
                            # Prepare input tensor: (1, sequence_length, n_channels, sensor_length)
                            sequence_data = np.stack(list(sequence_buffer), axis=0)  # (seq_len, 2, sensor_len)
                            
                            # Apply per-sequence normalization (matching training)
                            sequence_min = sequence_data.min()
                            sequence_max = sequence_data.max()
                            sequence_diff = sequence_max - sequence_min
                            
                            if sequence_diff > 0:
                                sequence_data = (sequence_data - sequence_min) / sequence_diff
                            
                            sequence_data = sequence_data[np.newaxis, ...]  # (1, seq_len, 2, sensor_len)
                            
                            # Convert to tensor and run inference
                            with torch.no_grad():
                                input_tensor = torch.FloatTensor(sequence_data).to(device)
                                start_time = time.time()
                                logits = model(input_tensor)
                                end_time = time.time()
                                inference_latency = end_time - start_time
                                inference_latencies.append(inference_latency)
                                print(f"Inference latency: {inference_latency * 1000:.2f} ms")
                                
                                # Apply center class bias to make center prediction easier
                                # Center class is at index 2, add bias to its logit
                                center_bias = 0.75  # Adjust this value to control bias strength
                                logits[:, 2] += center_bias  # Boost center class logits
                                
                                # Advanced center preference logic
                                # If center logit is within threshold of max logit, prefer center
                                center_preference_threshold = 0.1
                                max_logit = torch.max(logits)
                                center_logit = logits[:, 2]
                                
                                if max_logit - center_logit <= center_preference_threshold:
                                    # Center is close to max, force center prediction
                                    prediction = torch.tensor([2]).cpu().numpy()  # Force center class (index 2)
                                else:
                                    prediction = torch.argmax(logits, dim=1).cpu().numpy()
                            
                            # Add the prediction to the buffer for majority voting
                            buffer.append(prediction[0])
                            if len(buffer) > 5:
                                buffer.pop(0)

                            # Perform majority voting on the buffer
                            try:
                                predicted_class_index = statistics.mode(buffer)
                            except statistics.StatisticsError:
                                predicted_class_index = buffer[-1]
                            
                            # Prediction stability mechanism - prevent rapid transitions
                            current_time = time.time()
                            if (last_predicted_class_index is not None and 
                                predicted_class_index != last_predicted_class_index and
                                current_time - last_class_change_time < stable_prediction_duration):
                                # Too soon to change class, keep the previous prediction
                                predicted_class_index = last_predicted_class_index
                            
                            # Display prediction with class name
                            predicted_class_name = CLASS_NAMES[predicted_class_index]
                            print(f"Predicted: {predicted_class_name} (class {predicted_class_index})")

                            # Update image display and audio cue only if the class index changes
                            if predicted_class_index != last_predicted_class_index:
                                cv2.imshow('Predicted Class', class_images[predicted_class_index])
                                cv2.waitKey(1)  # Required to update the image display
                                
                                # Audio cue for the prediction (non-blocking)
                                play_audio_cue(predicted_class_name)
                                
                                last_predicted_class_index = predicted_class_index
                                last_class_change_time = current_time  # Update the time of last class change

            except ValueError as e:
                print(f"Invalid data format: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

except KeyboardInterrupt:
    print("Stopping...")
finally:
    ser.close()
    cv2.destroyAllWindows()
