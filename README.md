<h1>Local Vision-Driven Customer Traffic Analyser</h1>

<h2>Description:</h2>
An on-device, privacy-first computer vision solution designed for physical retail store environments. Running locally on an NVIDIA Jetson platform, the application uses advanced tracking algorithms to process live or pre-recorded camera feeds, mapping spatial visitor patterns into high-contrast density heatmaps. By processing data at the network edge, the system generates actionable spatial intelligence regarding product engagement and high-traffic floor bottlenecks without recording or storing customer video footage.

<img width="960" height="481" alt="Screenshot 2026-07-02 144855" src="https://github.com/user-attachments/assets/ce82cdf8-849d-4979-9647-92624e0f2a07" />


<h2>The Algorithm:</h2>
The application runs an analytical pipeline at the network edge to translate raw real-time video streams as well as uploaded footage into static spatial density data. It relies on object detection, object tracking, and image matrix manipulation, avoiding the need to store raw video files.

**Process Workflow:**

[ Camera/Video Feed ] ──> [ Frame-Skip Filter ] ──> [ YOLO11 Object Detection ] ──> [ Heatmap Image Generation ] ──> [ Matrix Aggregation ] ──> [ Object Tracking (IDs) ]



**Frame Ingestion & Filtering:**
To optimize resource utilization on the Nvidia Jetson hardware modules, incoming video frames pass through a calculated skip-interval filter (skip_step). Instead of pushing the full camera stream through the neural network, the app only processes every N-th frame (e.g., every 3rd or 5th frame) based on the user's sidebar settings.

**Object Detection (YOLO11):**
Filtered frames are processed by an optimized yolo11n.pt (nano) model running locally. The object detection engine isolates features down to Class 0 (Humans), drawing temporary bounding box matrices around every individual customer detected in the field of view.

**Object Tracking:**
Using the ultralytics.solutions.Heatmap framework, the algorithm tracks spatial persistence across subsequent frames. Each human detection is assigned a unique tracking identity tag (track_id). The system monitors individual paths over time to prevent double-counting static objects.
<img width="923" height="341" alt="Screenshot 2026-07-02 144952" src="https://github.com/user-attachments/assets/0c9e559a-27d5-435f-830b-354a0a41d0d0" />


**Spatial Matrix Aggregation:**
Instead of recording video, the program maps coordinates onto an internal 2D relative intensity matrix matching the stream's dimensions. As a tracked ID lingers over explicit (x, y) coordinate boundaries, the values at those matrix coordinates scale upward.

**Dynamic Background Isolation:**
The algorithm continuously checks frame states to track the least-populated state (min_people_count). It saves a clean, copy template of the background scene (best_background_frame), which serves as the canvas for the final visual layout.

**Mathematical Normalization & Color Overlays:**
When the tracking loop completes, the accumulated raw counter values undergo non-linear scaling (√x) to minimize noise spikes, followed by min-max normalization mapping the data to standard 8-bit image values (0-255):

<img width="986" height="396" alt="Screenshot 2026-07-02 145135" src="https://github.com/user-attachments/assets/f04e0322-d480-46e6-ad56-3d69f1545fcc" />





This normalized matrix passes through an alpha-blending filter (cv2.addWeighted), merging the structural background frame with a high-contrast thermal map (cv2.COLORMAP_JET). Low-frequency zones appear cool blue, while high-duration lingering zones render as high-visibility red hot spots.

**Core Dependencies:**
**Ultralytics:**
Loads the core vision model layers, manages inference execution, and isolates target classes.

**Opencv-python (cv2):**
Handles low-level video capturing (VideoCapture), camera endpoint connections, color channel conversions (BGR to RGB), matrix manipulation, and image writing operations.

**Streamlit:**
Manages the local presentation layer, updates user interface states, handles layout components, and isolates variables inside persistent session arrays.

**Numpy:**
Conducts rapid, vectorized mathematical operations on the spatial arrays and handles data type casting.


<h2>Running This Project:</h2>

Connect to the Jetson via SSH
`ssh username@<YOUR_JETSON_IP_ADDRESS>`

**Section 1: Physical Hardware & Kernel Node Mapping**
-------------------------------------------------------------------------------------------------------------------------------------
Before initializing the AI pipeline, the video capture hardware must be mounted and verified within the Linux kernel space.

Physical Camera Mount
Connect your UVC-compliant USB webcam or CSI camera module directly into one of the four high-speed USB Type-A ports located on the back of your NVIDIA Jetson carrier board. Ensure your developer kit is running in its maximum performance power mode.

Query Active Video Kernel Nodes
`ls -l /dev/video*`

Interpret Hardware Index Mapping
The primary video stream is typically bound to /dev/video0, which corresponds directly to hardware index 0 within the application logic loops. Additional indexes like /dev/video1 are metadata channels mapped by the OS kernel.

**Section 2: System Architecture & Library Installation**
-------------------------------------------------------------------------------------------------------------------------------------
This section configures the Jetson native development environment and installs the computer vision, deep learning, and user interface software packages.

Install System Compilers
`sudo apt update && sudo apt install -y build-essential python3-pip python3-dev`

Install Python Dependencies
`pip3 install ultralytics streamlit numpy opencv-python`

Verify Installation
`python3 -c "import ultralytics, streamlit, cv2; print('YOLO:', ultralytics.version); print('Streamlit:', streamlit.version)"`

**Section 3: Project Setup & Execution**
-------------------------------------------------------------------------------------------------------------------------------------
This section covers navigating to the workspace, creating the application script file, and launching the real-time analytics server.

Navigate to Project Directory
`cd retail_analytics`

Create the Application File and Paste Your Code
Before running the application, make sure you have pasted your project code into the file. If you haven't created it yet, open the file using a text editor (like nano) to paste your code inside:
nano app.py

Run the Analytics Application
`python3 -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501`

Access the Dashboard URL
Directly on the Jetson: `http://localhost:8501`
From another device on your local network:

**Section 4: Live Data Visualization & Code Maintenance**
-------------------------------------------------------------------------------------------------------------------------------------
This section covers managing the web interface layout, adjusting model settings via the interface, and securely shutting down the background processes.

Adjust Real-Time Tracking Thresholds
Modify the confidence slider directly on the left sidebar of the running interface to filter out weak object detections without restarting the background service.

Toggle Dashboard Performance Metrics
Check or uncheck the rendering telemetry options in the control panel to view frames per second (FPS) and tracking latency dynamically calculated on the Jetson hardware.

Gracefully Terminate the Application Server
When you are finished running the dashboard, click on your active terminal window and press the keyboard shortcut to kill the background Python process:
Ctrl+C


<h2>Video Showcase:</h2>
https://drive.google.com/file/d/1lu7_c1CNbKwWTtsVU7uxa5cNcEbNXbls/view?usp=drive_link 
