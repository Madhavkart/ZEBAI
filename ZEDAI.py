# Install necessary packages
!pip install supervision opencv-python matplotlib
!apt-get install python3-tk
!pip install python-docx
!pip install opencv-python-headless

# Import libraries
import requests
import base64
from PIL import Image
from io import BytesIO
import os
import supervision as sv
from tqdm import tqdm
from supervision import get_video_frames_generator
import time
from docx import Document
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# Constants
INFERENCE_ENDPOINT = "https://infer.roboflow.com"
API_KEY = "57ypj0AlMvyEJAlPlmXz"
VIDEO = "/content/input_vedio.mp4"

# Prompt list to evaluate similarity between each image and each prompt. If something else is selected, then we ignore the caption
# Change this to your desired prompt list
prompt_list = [['zebrafish', "something else"]]

def classify_image(image: str, prompt: str) -> dict:
    image_data = Image.fromarray(image)
    buffer = BytesIO()
    image_data.save(buffer, format="JPEG")
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    payload = {
        "api_key": '57ypj0AlMvyEJAlPlmXz',
        "subject": {
            "type": "base64",
            "value": image_data
        },
        "prompt": prompt,
    }

    data = requests.post(INFERENCE_ENDPOINT + "/clip/compare?api_key=" + '57ypj0AlMvyEJAlPlmXz', json=payload)
    response = data.json()
    sim = response["similarity"]

    highest_prediction = 0
    highest_prediction_index = 0

    for i, prediction in enumerate(response["similarity"]):
        if prediction > highest_prediction:
            highest_prediction = prediction
            highest_prediction_index = i

    return prompt[highest_prediction_index], sim[highest_prediction_index]

import torch
import torch.nn as nn

class YOLOv3(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Define the base convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # Define the detection layers
        self.det1 = nn.Conv2d(64, num_anchors * (num_classes + 5), kernel_size=1, stride=1, padding=0)
        self.det2 = nn.Conv2d(128, num_anchors * (num_classes + 5), kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        det1 = self.det1(x)
        det2 = self.det2(x)
        return det1, det2

def process_video(video_path):
    frames_generator = get_video_frames_generator(video_path)
    results = []

    for frame in tqdm(frames_generator, desc="Processing frames"):
        response = requests.post(INFERENCE_ENDPOINT, files={"file": frame})
        result = response.json()
        results.append(result)

    return results

def save_results_to_word(results, output_file):
    doc = Document()
    for result in results:
        doc.add_paragraph(str(result))
    doc.save(output_file)
    print(f"Document saved to {output_file}")

# New Functions for Tagging, Pathway Tracking, Speed Tracking, Object Counting, and Heatmap Generation

def tag_objects(frame, detections):
    for i, detection in enumerate(detections):
        x, y, w, h = detection['bbox']
        label = detection['label']
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        frame = cv2.putText(frame, f'{label} {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame

def track_object_pathway(results):
    pathways = {}
    for frame_idx, result in enumerate(results):
        for obj in result['objects']:
            obj_id = obj['id']
            if obj_id not in pathways:
                pathways[obj_id] = {'x': [], 'y': [], 'frame_idx': []}
            x, y, w, h = obj['bbox']
            pathways[obj_id]['x'].append(x + w // 2)
            pathways[obj_id]['y'].append(y + h // 2)
            pathways[obj_id]['frame_idx'].append(frame_idx)
    return pathways

def plot_pathways(pathways, output_file):
    plt.figure()
    for obj_id, pathway in pathways.items():
        plt.plot(pathway['x'], pathway['y'], label=f'Object {obj_id}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.title('Object Pathways')
    plt.savefig(output_file)
    plt.close()

def calculate_speeds(pathways, fps):
    speeds = {}
    for obj_id, pathway in pathways.items():
        speeds[obj_id] = []
        for i in range(1, len(pathway['frame_idx'])):
            dx = pathway['x'][i] - pathway['x'][i-1]
            dy = pathway['y'][i] - pathway['y'][i-1]
            dt = (pathway['frame_idx'][i] - pathway['frame_idx'][i-1]) / fps
            speed = np.sqrt(dx**2 + dy**2) / dt
            speeds[obj_id].append(speed)
    return speeds

def plot_speeds(speeds, output_file):
    plt.figure()
    for obj_id, speed in speeds.items():
        plt.plot(speed, label=f'Object {obj_id}')
    plt.xlabel('Frame Index')
    plt.ylabel('Speed (pixels/frame)')
    plt.legend()
    plt.title('Object Speeds')
    plt.savefig(output_file)
    plt.close()

def count_objects(results):
    counts = []
    for result in results:
        counts.append(len(result['objects']))
    return counts

def plot_object_counts(counts, output_file):
    plt.figure()
    plt.plot(counts)
    plt.xlabel('Frame Index')
    plt.ylabel('Number of Objects')
    plt.title('Object Counts per Frame')
    plt.savefig(output_file)
    plt.close()

def generate_heatmap(pathways, frame_shape, output_file):
    heatmap = np.zeros(frame_shape[:2])
    for pathway in pathways.values():
        for x, y in zip(pathway['x'], pathway['y']):
            heatmap[y, x] += 1
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Object Activity Heatmap')
    plt.savefig(output_file)
    plt.close()

# Updated function to insert user's video
def insert_your_video(video_path):
    if not os.path.exists(video_path):
        print("File not found. Please check the path and try again.")
        return
    
    results = process_video(video_path)
    
    # Create the output folder if it doesn't already exist
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Tagging objects
    frames_generator = get_video_frames_generator(video_path)
    for frame_idx, frame in enumerate(frames_generator):
        detections = results[frame_idx]['objects']
        frame = tag_objects(frame, detections)
        tagged_frame_path = os.path.join(output_folder, f'tagged_frame_{frame_idx}.jpg')
        cv2.imwrite(tagged_frame_path, frame)

    # Pathway tracking
    pathways = track_object_pathway(results)
    pathway_output_file = os.path.join(output_folder, "object_pathways.png")
    plot_pathways(pathways, pathway_output_file)

    # Speed tracking
    fps = 30  # Assuming a frame rate of 30 FPS
    speeds = calculate_speeds(pathways, fps)
    speed_output_file = os.path.join(output_folder, "object_speeds.png")
    plot_speeds(speeds, speed_output_file)

    # Object counting
    counts = count_objects(results)
    counts_output_file = os.path.join(output_folder, "object_counts.png")
    plot_object_counts(counts, counts_output_file)

    # Heatmap generation
    # Assuming the first frame's shape
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the video file.")
        return
    frame_shape = frame.shape
    heatmap_output_file = os.path.join(output_folder, "object_heatmap.png")
    generate_heatmap(pathways, frame_shape, heatmap_output_file)

    # Save results to a Word document
    output_file = os.path.join(output_folder, "object_analysis.docx")
    save_results_to_word(results, output_file)

# UI for the Software with Grey Theme
def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if file_path:
        video_path.set(file_path)

def run_analysis():
    video_file = video_path.get()
    if not video_file:
        messagebox.showwarning("Input Error", "Please select a video file.")
        return
    insert_your_video(video_file)
    messagebox.showinfo("Process Completed", "Object detection and analysis completed successfully.")

# Setting up the main window with grey theme
root = tk.Tk()
root.title("Object Detection and Analysis Software")

# Set grey theme
root.configure(bg='#2f2f2f')

video_path = tk.StringVar()

# Configure styles
style = ttk.Style()
style.configure("TLabel", background="#2f2f2f", foreground="#ffffff", font=("Helvetica", 12))
style.configure("TEntry", fieldbackground="#4f4f4f", foreground="#ffffff")
style.configure("TButton", background="#3f3f3f", foreground="#ffffff", font=("Helvetica", 12))
style.map("TButton", background=[('active', '#5f5f5f')])

# Add UI elements
ttk.Label(root, text="Select Video File:").grid(row=0, column=0, padx=10, pady=10)
ttk.Entry(root, textvariable=video_path, width=50).grid(row=0, column=1, padx=10, pady=10)
ttk.Button(root, text="Browse", command=select_video).grid(row=0, column=2, padx=10, pady=10)
ttk.Button(root, text="Run Analysis", command=run_analysis).grid(row=1, column=0, columnspan=3, pady=20)

root.mainloop()
