import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import os
import cv2
import tempfile
import time

# =====================
# Paths
# =====================
RUNS_DIR = Path("D:/EWU/10th Semester/CSE475/LABS/Project/Streamlit App/runs_ssl")
st.set_page_config(page_title="Sunflower SSL Dashboard", layout="wide")
st.title("🌻 Sunflower Image Detection in Real-Time Dashboard")

# =====================
# Sidebar
# =====================
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.subheader("🧪 Experiment Settings")

exp_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
custom_labels = [
    "BYOL-YOLOv10s", "BYOL-YOLOv11s", "BYOL-YOLOv12s",
    "PSEUDO-STACK-(10-90)", "PSEUDO-STACK-(20-80)", "PSEUDO-STACK-(30-70)",
    "PSEUDO-STACK-(40-60)",
    "DINO-YOLOv10s", "DINO-YOLOv11s", "DINO-YOLOv12s"
]
exp_choice_label = st.sidebar.selectbox("🌟 Choose SSL Experiment", custom_labels)
exp_choice = exp_dirs[custom_labels.index(exp_choice_label)]

best_model_path = exp_choice / "weights" / "best.pt"
results_csv = exp_choice / "results.csv"
results_png = exp_choice / "results.png"

st.sidebar.subheader("🤖 Model Configuration")
st.sidebar.markdown(f"**Experiment:** {exp_choice_label}")

st.sidebar.subheader("🔍 Prediction Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# Subsection for detection vs plots
st.sidebar.subheader("📂 Dashboard View")
view_choice = st.sidebar.radio("Choose View", ["Detections", "Plots", "Arm Control"])


# =====================
# Load model
# =====================
@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))

model = load_model(best_model_path)

# =====================
# Detections (Image/Video/Live Camera)
# =====================
if view_choice == "Detections":
    st.subheader("🔍 Test Your Model on a Custom Image/Video")
    st.markdown(f"**Using Model:** `{exp_choice_label}`")

    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg","jpeg","png","mp4"])
    preds_dir = exp_choice / "preds"
    os.makedirs(preds_dir, exist_ok=True)

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        temp_path = Path(tempfile.mkdtemp()) / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        if file_ext in ["jpg","jpeg","png"]:
            img = Image.open(temp_path).convert("RGB")
            results = model.predict(source=img, conf=conf_threshold, verbose=False)
            annotated_img = results[0].plot(line_width=1, font_size=5)

            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Uploaded Image", width=600)
            with col2:
                st.image(annotated_img[:, :, ::-1], caption="Annotated Result", width=600)

            save_path = preds_dir / uploaded_file.name
            Image.fromarray(annotated_img[:, :, ::-1]).save(save_path)
            with open(save_path, "rb") as f:
                st.download_button("💾 Download Annotated Image", f, file_name=uploaded_file.name, mime="image/png")

        elif file_ext == "mp4":
            cap = cv2.VideoCapture(str(temp_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            file_stem = Path(uploaded_file.name).stem
            annotated_video_path = preds_dir / f"annotated_{file_stem}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (width, height))

            frame_count = 0
            progress_bar = st.progress(0)
            progress_text = st.empty()
            frame_placeholder = st.empty()
            start_time = time.time()

            with st.spinner("Processing video and generating annotations..."):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(frame, conf=conf_threshold, verbose=False)
                    annotated_frame = results[0].plot(line_width=1, font_size=4)
                    out.write(annotated_frame[:, :, ::-1])

                    frame_placeholder.image(annotated_frame[:, :, ::-1], channels="RGB", width=1000)

                    frame_count += 1
                    if total_frames > 0:
                        progress = int((frame_count / total_frames) * 100)
                        progress_bar.progress(progress)
                        elapsed = time.time() - start_time
                        remaining = max(0, int((elapsed / frame_count) * (total_frames - frame_count)))
                        progress_text.markdown(
                            f"Processing frame {frame_count}/{total_frames} | Estimated time left: {remaining}s"
                        )

            cap.release()
            out.release()
            progress_bar.empty()
            progress_text.empty()
            frame_placeholder.empty()
            st.success("✅ Video annotation complete!")

            with open(annotated_video_path, "rb") as f:
                st.download_button(
                    "💾 Download Annotated Video",
                    f,
                    file_name=f"annotated_{file_stem}.mp4",
                    mime="video/mp4"
                )

    # Live Camera
    st.subheader("📷 Live Camera Detection (Target: 30 FPS)")

    st.sidebar.subheader("🎥 Camera Source")
    camera_type = st.sidebar.radio(
        "Select Camera",
        ["Laptop Webcam", "External USB Webcam", "Mobile Camera (IP/RTSP)"]
    )

    mobile_url = None
    if camera_type == "Mobile Camera (IP/RTSP)":
        mobile_url = st.sidebar.text_input(
            "Enter Camera URL (e.g., http://192.168.0.101:8080/video)",
            value="http://192.168.0.101:8080/video"
        )
        if st.sidebar.button("🔍 Test Camera URL"):
            cap_test = cv2.VideoCapture(mobile_url)
            ret, frame = cap_test.read()
            if ret:
                st.sidebar.success("✅ Camera is reachable!")
                st.sidebar.image(frame[:, :, ::-1], channels="RGB", width=300)
            else:
                st.sidebar.error("❌ Could not reach the camera. Check URL or network.")
            cap_test.release()

    start_cam = st.button("▶️ Start Live Camera")

    if start_cam:
        if camera_type == "Laptop Webcam":
            cap = cv2.VideoCapture(0)
        elif camera_type == "External USB Webcam":
            cap = cv2.VideoCapture(1)
        elif camera_type == "Mobile Camera (IP/RTSP)" and mobile_url:
            cap = cv2.VideoCapture(mobile_url)
        else:
            st.error("❌ Please provide a valid camera URL or device")
            cap = None

        if cap and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            col1, col2 = st.columns([2, 1])
            with col1:
                frame_placeholder = st.empty()
            with col2:
                stop_cam = st.button("⏹️ Stop Camera")
                fps_text = st.empty()

            prev_time = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Could not read frame. Check camera or URL.")
                    break

                frame = cv2.resize(frame, (640, 480))
                results = model(frame, stream=True, conf=conf_threshold, verbose=False)
                for r in results:
                    annotated_frame = r.plot(line_width=2, font_size=12)

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time

                frame_placeholder.image(
                    annotated_frame[:, :, ::-1], channels="RGB", width=600
                )
                fps_text.markdown(f"**FPS:** {fps:.1f}")

                elapsed = time.time() - curr_time
                wait_time = max(0, (1/30) - elapsed)
                time.sleep(wait_time)

                if stop_cam:
                    break

            cap.release()
            st.success("✅ Camera stopped")

# =====================
# Plots (Training Performance)
# =====================
elif view_choice == "Plots":
    st.subheader("📈 Training Performance Overview")
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        small_figsize = (8,4)

        with col1:
            fig, ax = plt.subplots(figsize=small_figsize)
            if "epoch" in df.columns and "train/box_loss" in df.columns and "val/box_loss" in df.columns:
                ax.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
                ax.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss")
                ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
                ax.set_title("Train vs Val Box Loss")
                ax.legend(); ax.grid(True)
                st.pyplot(fig)

        with col2:
            loss_cols = ["epoch", "train/box_loss","train/cls_loss","val/box_loss","val/cls_loss"]
            if all(col in df.columns for col in loss_cols):
                fig2, ax2 = plt.subplots(figsize=small_figsize)
                ax2.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", linewidth=2)
                ax2.plot(df["epoch"], df["train/cls_loss"], label="Train Class Loss", linewidth=2)
                ax2.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss", linestyle="--")
                ax2.plot(df["epoch"], df["val/cls_loss"], label="Val Class Loss", linestyle="--")
                ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.set_title("Detailed Train/Val Loss")
                ax2.legend(); ax2.grid(True)
                st.pyplot(fig2)

        with col3:
            lr_columns = [col for col in df.columns if col.startswith("lr/pg")]
            if lr_columns:
                fig3, ax3 = plt.subplots(figsize=small_figsize)
                colors = ['purple','orange','green']; styles=['-','--','-.']; markers=['o','s','^']
                for idx,col in enumerate(lr_columns):
                    ax3.plot(df["epoch"], df[col], label=col, color=colors[idx%3], linestyle=styles[idx%3],
                            marker=markers[idx%3], markersize=4, linewidth=2)
                ax3.set_xlabel("Epoch"); ax3.set_ylabel("Learning Rate"); ax3.set_title("Learning Rates")
                ax3.legend(); ax3.grid(True)
                st.pyplot(fig3)

        with col4:
            map_keys = ["metrics/mAP50(B)","metrics/mAP50-95(B)"]
            map_col = next((k for k in map_keys if k in df.columns), None)
            if map_col:
                fig4, ax4 = plt.subplots(figsize=small_figsize)
                ax4.plot(df["epoch"], df[map_col], label=map_col, color="green", linewidth=2)
                ax4.set_xlabel("Epoch"); ax4.set_ylabel("mAP"); ax4.set_title("mAP During Training")
                ax4.grid(True); ax4.legend(); st.pyplot(fig4)

    st.subheader("🖼️ Results Image")
    if results_png.exists():
        img = Image.open(results_png)
        st.image(img, caption="Results Image", use_container_width=True)

elif view_choice == "Arm Control":
    st.subheader("🤖 xArm 1S Control Dashboard")

    import xarm
    import cv2
    import time
    from pathlib import Path
    import os

    control_mode = st.radio(
        "Select Control Mode",
        ["Manual Control", "Automated YOLO Pick-and-Place"]
    )

    # =====================
    # Shared Camera Setup
    # =====================
    st.sidebar.subheader("🎥 Camera Source")
    camera_type = st.sidebar.radio(
        "Select Camera",
        ["Laptop Webcam", "External USB Webcam", "Mobile Camera (IP/RTSP)"],
        key="arm_camera_type"
    )

    mobile_url = None
    if camera_type == "Mobile Camera (IP/RTSP)":
        mobile_url = st.sidebar.text_input(
            "Enter Camera URL (e.g., http://192.168.0.101:8080/video)",
            value="http://192.168.0.101:8080/video",
            key="arm_camera_url"
        )
        if st.sidebar.button("🔍 Test Camera URL"):
            cap_test = cv2.VideoCapture(mobile_url)
            ret, frame = cap_test.read()
            if ret:
                st.sidebar.success("✅ Camera is reachable!")
                st.sidebar.image(frame[:, :, ::-1], channels="RGB", width=300)
            else:
                st.sidebar.error("❌ Could not reach the camera. Check URL or network.")
            cap_test.release()

    # =====================
    # Manual Control
    # =====================
    if control_mode == "Manual Control":
        class XArmManualGUI:
            def __init__(self):
                self.robot = self.connect_to_robot()
                if self.robot is None:
                    st.error("Exiting application due to connection failure.")
                    st.stop()
                st.success("✅ xArm connected successfully")

                default_value = 500
                keys = [
                    "gripper","link2","link3","link4","link5","link6",
                    "torque_gripper","torque_link2","torque_link3",
                    "torque_link4","torque_link5","torque_link6"
                ]
                for key in keys:
                    st.session_state.setdefault(key, default_value)

                # Layout
                col1, col2, col3 = st.columns([2,2,1])
                with col3:
                    if st.button("🔄 Reset to Default"):
                        for key in keys:
                            st.session_state[key] = default_value
                        st.rerun()
                        st.info("✅ Sliders reset to default")

                with col1:
                    st.slider("Gripper Position", 0, 1000, st.session_state.gripper, key="gripper")
                    st.slider("Link 2 Position", 0, 1000, st.session_state.link2, key="link2")
                    st.slider("Link 3 Position", 0, 1000, st.session_state.link3, key="link3")
                    st.slider("Link 4 Position", 0, 1000, st.session_state.link4, key="link4")
                    st.slider("Link 5 Position", 0, 1000, st.session_state.link5, key="link5")
                    st.slider("Link 6 Position", 0, 1000, st.session_state.link6, key="link6")

                with col2:
                    st.slider("Gripper Torque", 0, 1000, st.session_state.torque_gripper, key="torque_gripper")
                    st.slider("Link 2 Torque", 0, 1000, st.session_state.torque_link2, key="torque_link2")
                    st.slider("Link 3 Torque", 0, 1000, st.session_state.torque_link3, key="torque_link3")
                    st.slider("Link 4 Torque", 0, 1000, st.session_state.torque_link4, key="torque_link4")
                    st.slider("Link 5 Torque", 0, 1000, st.session_state.torque_link5, key="torque_link5")
                    st.slider("Link 6 Torque", 0, 1000, st.session_state.torque_link6, key="torque_link6")

                self.update_arm()

            def update_arm(self):
                positions = [
                    st.session_state.gripper, st.session_state.link2,
                    st.session_state.link3, st.session_state.link4,
                    st.session_state.link5, st.session_state.link6
                ]
                torques = [
                    st.session_state.torque_gripper, st.session_state.torque_link2,
                    st.session_state.torque_link3, st.session_state.torque_link4,
                    st.session_state.torque_link5, st.session_state.torque_link6
                ]
                min_duration, max_duration = 0.05, 2.0
                for i, (pos, tq) in enumerate(zip(positions, torques), start=1):
                    duration = max_duration - ((tq / 1000) * (max_duration - min_duration))
                    self.robot.setPosition(i, pos, int(duration*1000), wait=False)

            def connect_to_robot(self):
                try:
                    robot = xarm.Controller('USB')
                    st.info("Connected to xArm robot")
                    return robot
                except:
                    st.error("Failed to connect to xArm robot")
                    return None

        gui = XArmManualGUI()

    # =====================
    # Automated YOLO Tracking (Camera attached to Arm)
    # =====================
    elif control_mode == "Automated YOLO Pick-and-Place":
        class XArmAutoGUI:
            def __init__(self):
                self.robot = self.connect_to_robot()
                if self.robot is None:
                    st.error("Exiting due to connection failure.")
                    st.stop()
                st.success("✅ xArm connected successfully")

                # Camera attached to xArm
                self.cap = cv2.VideoCapture(2)  # Adjust camera ID
                if not self.cap.isOpened():
                    st.error("❌ Could not open the xArm-attached camera")
                    st.stop()
                
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

                # Video saving
                self.preds_dir = Path(exp_choice) / "preds"
                os.makedirs(self.preds_dir, exist_ok=True)
                self.save_path = self.preds_dir / "automated_arm_tracking.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out = cv2.VideoWriter(str(self.save_path), fourcc, 20, (640, 480))

                # Streamlit placeholders
                self.frame_placeholder = st.empty()
                self.fps_text = st.empty()
                self.stop_button = st.button("⏹️ Stop Tracking Session")

                # Default positions & torque
                self.positions = [500]*6
                self.torques = [500]*6
                self.target_height = 400  # Z-axis height to follow objects

                self.run_tracking()

            def run_tracking(self):
                st.write("### Automated Tracking Running...")
                prev_time = time.time()
                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        st.warning("⚠️ Cannot read frame")
                        break

                    frame = cv2.resize(frame, (640, 480))
                    results = model(frame, stream=True, conf=conf_threshold, verbose=False)
                    annotated_frame = frame.copy()
                    object_center = None

                    # Detect first object (or largest)
                    for r in results:
                        annotated_frame = r.plot(line_width=2, font_size=12)
                        if len(r.boxes.xyxy) > 0:
                            x1, y1, x2, y2 = map(int, r.boxes.xyxy[0])
                            object_center = ((x1 + x2)//2, (y1 + y2)//2)
                            break

                    # Move arm to follow the object
                    if object_center:
                        cx, cy = object_center
                        # Map pixel to arm workspace (example: 0-640 -> 0-800 units)
                        servo_x = int(cx / 640 * 800)
                        servo_y = int(cy / 480 * 800)
                        self.positions[:2] = [servo_x, servo_y]
                        self.positions[2] = self.target_height  # maintain Z height
                        self.send_val()

                    # Save video
                    self.out.write(cv2.resize(annotated_frame, (640, 480)))

                    # Display frame and FPS
                    self.frame_placeholder.image(annotated_frame[:, :, ::-1], channels="RGB", width=600)
                    curr_time = time.time()
                    fps = 1 / max(curr_time - prev_time, 0.001)
                    prev_time = curr_time
                    self.fps_text.markdown(f"**FPS:** {fps:.1f}")

                    if self.stop_button:
                        break
                    time.sleep(0.05)

                self.cap.release()
                self.out.release()
                st.success(f"✅ Tracking session saved at `{self.save_path}`")
                with open(self.save_path, "rb") as f:
                    st.download_button("💾 Download Tracking Video", f, file_name=self.save_path.name, mime="video/mp4")

            def send_val(self):
                min_d, max_d = 0.05, 2.0
                for i, (pos, tq) in enumerate(zip(self.positions, self.torques), start=1):
                    duration = max_d - ((tq / 1000) * (max_d - min_d))
                    self.robot.setPosition(i, pos, int(duration*1000), wait=False)

            def connect_to_robot(self):
                try:
                    robot = xarm.Controller('USB')
                    st.info("Connected to xArm robot")
                    return robot
                except:
                    st.error("Failed to connect to xArm robot")
                    return None

        gui = XArmAutoGUI()
