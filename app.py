import streamlit as st
import cv2
import os
import logging
import numpy as np
import time
from datetime import datetime
import glob
from ultralytics import solutions

# Keep the console outputs clear of unnecessary model telemetry warnings
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# --- Page Configuration ---
st.set_page_config(page_title="Local Retail Traffic Analyser", layout="wide")

# --- Define Safe System Local Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "Videos")
LIBRARY_DIR = os.path.join(BASE_DIR, "Library")

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(LIBRARY_DIR, exist_ok=True)

# Default dynamic naming template function
def generate_heatmap_path():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(LIBRARY_DIR, f"heatmap_{timestamp}.png")

# --- Initialize Session States ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.60  # Ideal threshold from testing
if 'frame_rate' not in st.session_state:
    st.session_state.frame_rate = "Every Frame"

# --- Sidebar: System Configuration Dials ---
with st.sidebar:
    st.markdown("###  Model Settings")
    st.session_state.confidence = st.slider(
        "Human Detection Confidence Threshold", 
        min_value=0.1, max_value=1.0, value=st.session_state.confidence, step=0.05
    )
    st.session_state.frame_rate = st.selectbox(
        "Analysis Frame Interval", 
        ["Every Frame", "Every 3rd Frame", "Every 5th Frame", "Every 10th Frame"],
        index=["Every Frame", "Every 3rd Frame", "Every 5th Frame", "Every 10th Frame"].index(st.session_state.frame_rate)
    )
    st.caption("Settings scale down processing workload on your Jetson hardware modules natively.")

def go_home():
    # If a live camera handle exists, make sure to release it back to the kernel before switching screens
    if 'live_cap_handle' in st.session_state and st.session_state.live_cap_handle is not None:
        st.session_state.live_cap_handle.release()
        st.session_state.live_cap_handle = None
    st.session_state.page = 'home'

# Compute explicit frames skipped based on selection
frame_skip_mapping = {
    "Every Frame": 1,
    "Every 3rd Frame": 3,
    "Every 5th Frame": 5,
    "Every 10th Frame": 10
}
skip_step = frame_skip_mapping[st.session_state.frame_rate]

# --- SCREEN 1: MAIN MENU ---
if st.session_state.page == 'home':
    st.title("Local Vision-Driven Customer Traffic Analyser")
    st.subheader("Privacy-First Edge AI Analytics for your Business")
    st.write("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(" Process Uploaded Video Footage", use_container_width=True, type="primary"):
            st.session_state.page = 'upload_mode'
            st.rerun()
            
    with col2:
        if st.button(" Analyze Live Camera Feed", use_container_width=True, type="primary"):
            st.session_state.page = 'live_mode'
            st.rerun()

    with col3:
        if st.button(" View Analytics Library", use_container_width=True, type="secondary"):
            st.session_state.page = 'library_mode'
            st.rerun()

# --- SCREEN 2A: UPLOAD FOOTAGE MODE ---
elif st.session_state.page == 'upload_mode':
    st.title("Upload Recorded Video")
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        temp_file_path = os.path.join(VIDEO_DIR, "temp_video.mp4")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success("Footage loaded and buffered locally to system hardware.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(" Start Analysis", use_container_width=True, type="primary"):
                
                cap = cv2.VideoCapture(temp_file_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                heatmap_manager = solutions.Heatmap(
                    model="yolo11n.pt",
                    colormap=cv2.COLORMAP_JET,
                    show=False,
                    verbose=False,
                    conf=st.session_state.confidence,
                    classes=[0]  # Mandate strictly humans (Class 0)
                )
                
                best_background_frame = None
                min_people_count = float('inf')
                frame_count = 0
                
                progress_bar = st.progress(0, text="AI core engine initializing layer allocations...")
                
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    frame_count += 1
                    if frame_count % skip_step != 0:
                        continue
                    
                    heatmap_manager.process(frame)
                    
                    current_people_count = 0
                    if hasattr(heatmap_manager, 'track_ids') and heatmap_manager.track_ids is not None:
                        current_people_count = len(heatmap_manager.track_ids)
                    
                    if current_people_count < min_people_count or best_background_frame is None:
                        min_people_count = current_people_count
                        best_background_frame = frame.copy()
                        
                    percent = min(int((frame_count / total_frames) * 100), 100)
                    progress_bar.progress(percent, text=f"Processing Video Frames: {percent}% ({frame_count}/{total_frames})")
                
                cap.release()
                
                if best_background_frame is not None and hasattr(heatmap_manager, 'heatmap') and heatmap_manager.heatmap is not None:
                    raw_heatmap = heatmap_manager.heatmap
                    if len(raw_heatmap.shape) == 3:
                        raw_heatmap = cv2.cvtColor(raw_heatmap, cv2.COLOR_BGR2GRAY)
                        
                    scaled_heatmap = np.sqrt(raw_heatmap)
                    heatmap_normalized = cv2.normalize(scaled_heatmap, None, 0, 255, cv2.NORM_MINMAX)
                    heatmap_8bit = heatmap_normalized.astype(np.uint8)
                    colored_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
                    final_overlay = cv2.addWeighted(best_background_frame, 0.5, colored_heatmap, 0.5, 0)
                    
                    target_path = generate_heatmap_path()
                    cv2.imwrite(target_path, final_overlay)
                    st.session_state.last_saved_heatmap = target_path
                    
                st.session_state.page = 'show_heatmap'
                st.rerun()
                
        with col2:
            if st.button(" Quit to Home", use_container_width=True):
                go_home()
                st.rerun()
    else:
        if st.button(" Back to Home", use_container_width=True):
            go_home()
            st.rerun()

# --- SCREEN 2B: LIVE CAMERA FEED MODE ---
elif st.session_state.page == 'live_mode':
    st.title("Live Camera Feed Processing")
    
    # Persistent Session-State Check: Check if camera initialization already occurred
    if 'live_cap_handle' not in st.session_state or st.session_state.live_cap_handle is None:
        cap = None
        active_index = -1
        for index in [0, 1]:  # Port 0 is confirmed by your system diagnostic
            test_cap = cv2.VideoCapture(index)
            if test_cap.isOpened():
                success, _ = test_cap.read()
                if success:
                    cap = test_cap
                    active_index = index
                    break
            test_cap.release()
            
        if cap is not None:
            st.session_state.live_cap_handle = cap
            st.session_state.live_cap_index = active_index

    # Error UI validation mapping
    if 'live_cap_handle' not in st.session_state or st.session_state.live_cap_handle is None:
        st.error(" Hardware Isolation Error: Camera at /dev/video0 is present but busy. Try refreshing your webpage browser layout to sync the thread.")
        if st.button(" Return to Main Menu", use_container_width=False):
            go_home()
            st.rerun()
    else:
        cap = st.session_state.live_cap_handle
        active_index = st.session_state.live_cap_index
        st.success(f" Connected to Jetson Hardware Camera Stream at index: /dev/video{active_index}")
        
        frame_placeholder = st.empty()
        
        # Initialize heatmap structure outside the running session context
        if 'live_heatmap_obj' not in st.session_state:
            st.session_state.live_heatmap_obj = solutions.Heatmap(
                model="yolo11n.pt",
                colormap=cv2.COLORMAP_JET,
                show=False,
                verbose=False,
                conf=st.session_state.confidence,
                classes=[0]
            )
        heatmap_manager = st.session_state.live_heatmap_obj
        
        if 'best_bg_frame' not in st.session_state:
            st.session_state.best_bg_frame = None
            st.session_state.min_peeps = float('inf')
        
        stop_clicked = st.button(" Stop Live Feed & Render Heatmap", type="primary", use_container_width=True)
        
        frame_count = 0
        while cap.isOpened() and not stop_clicked:
            success, frame = cap.read()
            if not success:
                break
                
            frame_count += 1
            
            if frame_count % skip_step == 0:
                heatmap_manager.process(frame)
                
                current_people_count = 0
                if hasattr(heatmap_manager, 'track_ids') and heatmap_manager.track_ids is not None:
                    current_people_count = len(heatmap_manager.track_ids)
                
                if current_people_count < st.session_state.min_peeps or st.session_state.best_bg_frame is None:
                    st.session_state.min_peeps = current_people_count
                    st.session_state.best_bg_frame = frame.copy()
            
            # Repaint from BGR to RGB for browser visualization
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            time.sleep(0.005)
            
        if stop_clicked:
            # Force release of the video interface capture back to the Linux Kernel
            cap.release()
            st.session_state.live_cap_handle = None
            
            progress_bar = st.progress(0, text="Consolidating telemetry arrays into layout matrix...")
            for percent in range(1, 101):
                time.sleep(0.005)
                progress_bar.progress(percent)
                
            # SAFETY CHECK: Extract structural frame mappings from storage limits
            if st.session_state.best_bg_frame is not None and hasattr(heatmap_manager, 'heatmap') and heatmap_manager.heatmap is not None:
                raw_heatmap = heatmap_manager.heatmap
                if len(raw_heatmap.shape) == 3:
                    raw_heatmap = cv2.cvtColor(raw_heatmap, cv2.COLOR_BGR2GRAY)
                    
                scaled_heatmap = np.sqrt(raw_heatmap)
                heatmap_normalized = cv2.normalize(scaled_heatmap, None, 0, 255, cv2.NORM_MINMAX)
                heatmap_8bit = heatmap_normalized.astype(np.uint8)
                colored_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
                final_overlay = cv2.addWeighted(st.session_state.best_bg_frame, 0.5, colored_heatmap, 0.5, 0)
                
                target_path = generate_heatmap_path()
                cv2.imwrite(target_path, final_overlay)
                st.session_state.last_saved_heatmap = target_path
                
                # Clear tracking cache arrays before changing state layout locations
                if 'live_heatmap_obj' in st.session_state:
                    del st.session_state.live_heatmap_obj
                if 'best_bg_frame' in st.session_state:
                    del st.session_state.best_bg_frame
                    
                st.session_state.page = 'show_heatmap'
                st.rerun()
            else:
                st.error(" The live feed stopped before generating a map. Let the camera track data brief moments longer.")
                if 'live_heatmap_obj' in st.session_state:
                    del st.session_state.live_heatmap_obj
                if 'best_bg_frame' in st.session_state:
                    del st.session_state.best_bg_frame
                if st.button(" Return to Main Menu"):
                    go_home()
                    st.rerun()

# --- SCREEN 2C: ANALYTICS LIBRARY VIEW MODE ---
elif st.session_state.page == 'library_mode':
    st.title(" On-Device Analytics Heatmap Library")
    st.write("Browse historical telemetry map layouts safely saved on your local Jetson storage layers.")
    
    if st.button(" Return to Main Menu", type="secondary"):
        go_home()
        st.rerun()
        
    st.write("---")
    
    # Grab all historical images inside the Library folder, sorted latest-first
    saved_heatmaps = sorted(glob.glob(os.path.join(LIBRARY_DIR, "heatmap_*.png")), reverse=True)
    
    if not saved_heatmaps:
        st.info("No saved heatmaps found in the local database. Run an video or camera sequence analysis first!")
    else:
        # Create a scannable grid layout with 3 columns
        cols = st.columns(3)
        for index, path in enumerate(saved_heatmaps):
            current_col = cols[index % 3]
            
            # Format filename strings back into clean titles
            filename = os.path.basename(path)
            raw_timestamp = filename.replace("heatmap_", "").replace(".png", "")
            try:
                parsed_time = datetime.strptime(raw_timestamp, "%Y%m%d_%H%M%S")
                clean_date = parsed_time.strftime(" %B %d, %Y  |   %I:%M:%S %p")
            except ValueError:
                clean_date = f"File: {filename}"
                
            with current_col:
                img = cv2.imread(path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Render inside a clean, expandable panel container
                with st.container(border=True):
                    st.image(rgb_img, use_container_width=True)
                    st.caption(f"**{clean_date}**")
                    
                    # Provide a targeted expander panel to review full detail sizes
                    with st.expander(" View Full Scale Matrix"):
                        st.image(rgb_img, use_container_width=True)

# --- SCREEN 3: OUTPUT RESULT (HEATMAP) ---
elif st.session_state.page == 'show_heatmap':
    st.title(" Analysis Results: Customer Traffic Heatmap")
    st.write(f"Processed using **{st.session_state.frame_rate}** steps at an AI baseline confidence floor of **{int(st.session_state.confidence*100)}%**.")
    
    target_path = st.session_state.get('last_saved_heatmap', '')
    
    if target_path and os.path.exists(target_path):
        heatmap_img = cv2.imread(target_path)
        heatmap_rgb = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
        st.image(heatmap_rgb, caption=f"Density matrix complete. Successfully cataloged to: {os.path.basename(target_path)}", use_container_width=True)
    else:
        st.error("Error reading generated file layout assets.")
        
    st.success("Edge calculations complete. Processing kept fully local to device storage layers.")
    
    st.write("") 
    if st.button(" Back to Main Menu", type="secondary", use_container_width=False):
        go_home()
        st.rerun()
