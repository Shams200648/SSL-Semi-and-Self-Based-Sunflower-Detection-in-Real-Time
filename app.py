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
    st.subheader("🤖 xArm 1S Manual Control")

    import serial
    import serial.tools.list_ports

    # -----------------------------
    # Detect available COM ports
    # -----------------------------
    available_ports = [p.device for p in serial.tools.list_ports.comports()]
    if not available_ports:
        st.warning("⚠️ No COM ports detected. Connect your xArm and refresh.")
        st.stop()

    port_choice = st.selectbox("Select COM Port", available_ports)

    # -----------------------------
    # Connect to xArm
    # -----------------------------
    if "arm" not in st.session_state or st.session_state.get("port") != port_choice:
        from xarm_driver import XArm1S  # your driver class
        try:
            st.session_state.arm = XArm1S(port=port_choice, baudrate=115200)
            st.session_state.port = port_choice
            st.session_state.torque_enabled = {sid: True for sid in range(1,7)}  # track torque
            st.success(f"✅ Connected to xArm 1S on {port_choice}")
        except Exception as e:
            st.error(f"❌ Could not connect to {port_choice}: {e}")
            st.stop()

    arm = st.session_state.arm

    # -----------------------------
    # Real-time servo control
    # -----------------------------
    st.subheader("🎚️ Real-Time Joint Control")
    cols = st.columns(6)
    servo_angles = {}
    for i in range(6):
        with cols[i]:
            servo_angles[i+1] = st.slider(f"Servo {i+1}", 0, 180, 90, key=f"servo_{i+1}")
            arm.move_servo(i+1, servo_angles[i+1], duration=200)

    # -----------------------------
    # Move all servos together
    # -----------------------------
    st.subheader("⚡ Move All Servos Together")
    duration = st.slider("Duration (ms)", 100, 3000, 500, step=50)
    if st.button("Move All Servos"):
        arm.move_servos(servo_angles, duration)
        st.success(f"Moved all servos with duration {duration} ms")

    # -----------------------------
    # Torque control
    # -----------------------------
    st.subheader("🔧 Torque Control")
    torque_col1, torque_col2 = st.columns(2)
    with torque_col1:
        torque_servo = st.selectbox("Select Servo", [1,2,3,4,5,6])
    with torque_col2:
        if st.button("Enable Torque"):
            arm.set_torque(torque_servo, True)
            st.session_state.torque_enabled[torque_servo] = True
            st.info(f"Torque enabled for Servo {torque_servo}")
        if st.button("Disable Torque"):
            arm.set_torque(torque_servo, False)
            st.session_state.torque_enabled[torque_servo] = False
            st.warning(f"Torque disabled for Servo {torque_servo}")

    # -----------------------------
    # Home Position
    # -----------------------------
    st.subheader("🏠 Home Position")
    if st.button("Move All Servos to 90°"):
        # Enable torque for all if not already
        for sid in range(1,7):
            if not st.session_state.torque_enabled[sid]:
                arm.set_torque(sid, True)
                st.session_state.torque_enabled[sid] = True
        # Move all servos to 90°
        home_angles = {sid: 90 for sid in range(1,7)}
        arm.move_servos(home_angles, duration=500)
        st.success("✅ All servos moved to Home Position (90°)")

    # -----------------------------
    # Disconnect arm
    # -----------------------------
    if st.button("Shutdown Connection"):
        arm.close()
        del st.session_state.arm
        del st.session_state.port
        del st.session_state.torque_enabled
        st.warning("🔌 Disconnected from xArm")
