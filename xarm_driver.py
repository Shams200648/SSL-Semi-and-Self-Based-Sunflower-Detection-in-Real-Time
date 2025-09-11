import streamlit as st
import serial
import time

# -----------------------------
# COM port input
# -----------------------------
COM_PORT = st.text_input("Enter COM Port (e.g., COM3):", "COM3")
BAUDRATE = 115200

# -----------------------------
# Connect to XArm
# -----------------------------
if st.button("Connect"):
    try:
        ser = serial.Serial(COM_PORT, BAUDRATE, timeout=1)
        time.sleep(2)  # allow arm to initialize
        st.session_state.ser = ser
        st.success(f"✅ Connected to XArm 1S on {COM_PORT}")
    except Exception as e:
        st.error(f"❌ Could not connect: {e}")

# -----------------------------
# Function to move servo
# -----------------------------
def move_servo(servo_id, angle, duration=200):
    """Send command to move one servo"""
    pwm = int(500 + (angle / 180) * 2000)
    cmd = f"#{servo_id}P{pwm}T{duration}!\r\n"
    st.session_state.ser.write(cmd.encode("utf-8"))
    time.sleep(duration / 1000 + 0.05)

def move_all_servos(angles_dict, duration=200):
    """Move all servos at once"""
    cmd_parts = [f"#{sid}P{int(500 + (angle/180)*2000)}" for sid, angle in angles_dict.items()]
    cmd_str = "".join(cmd_parts) + f"T{duration}!\r\n"
    st.session_state.ser.write(cmd_str.encode("utf-8"))
    time.sleep(duration / 1000 + 0.05)

# -----------------------------
# Manual 6-Servo Control
# -----------------------------
if "ser" in st.session_state:
    st.subheader("🎚️ Manual Control for 6 Servos")
    
    cols = st.columns(6)
    servo_angles = {}
    
    for i in range(6):
        with cols[i]:
            angle = st.slider(f"Servo {i+1}", 0, 180, 90, key=f"servo_{i+1}")
            servo_angles[i+1] = angle
            move_servo(i+1, angle, duration=100)  # move immediately

    st.info("Move sliders to control each servo in real-time")

    # -----------------------------
    # Home Position Button
    # -----------------------------
    if st.button("🏠 Home Position (All 90°)"):
        home_angles = {sid: 90 for sid in range(1,7)}
        move_all_servos(home_angles, duration=500)
        st.success("✅ All servos moved to Home Position (90°)")

# -----------------------------
# Disconnect
# -----------------------------
if "ser" in st.session_state and st.button("Disconnect"):
    st.session_state.ser.close()
    del st.session_state.ser
    st.warning("🔌 Disconnected from XArm")
