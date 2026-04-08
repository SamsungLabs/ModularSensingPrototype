#!/usr/bin/env python3
"""
Helper script to generate audio WAV files for eye gaze classes
Run this script once to create the audio files needed for the demo
"""

import pyttsx3
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = SCRIPT_DIR

# Create audio directory if it doesn't exist
os.makedirs(AUDIO_DIR, exist_ok=True)

# Define the classes and their audio text
CLASS_NAMES = ['Top Right', 'Bottom Left', 'Bottom Right', 'Center', 'Top Left']
audio_texts = {name: name for name in CLASS_NAMES}

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 180)  # Speed of speech
engine.setProperty('volume', 1.0)  # Volume level

print("Generating audio files...")

for class_name, text in audio_texts.items():
    output_file = os.path.join(AUDIO_DIR, f"{class_name}.wav")
    
    try:
        # Save to file
        engine.save_to_file(text, output_file)
        engine.runAndWait()
        print(f"Created: {output_file}")
    except Exception as e:
        print(f"Error creating {output_file}: {e}")

print("\nAudio files generated successfully!")
print(f"Location: {AUDIO_DIR}")
print("\nYou can now run the main demo script.")
