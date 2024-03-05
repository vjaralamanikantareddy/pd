from flask import Flask, Response, render_template
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Flag to indicate if pose detection is active
pose_detection_active = False

# Function to capture video from the webcam
def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if pose_detection_active:
                frame = detect_pose(frame, pose)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route to render the HTML template
@app.route('/')
def index():
    return render_template('index.html', pose_detection_active=pose_detection_active)

# Route to start pose detection
@app.route('/start_pose_detection')
def start_pose_detection():
    global pose_detection_active
    pose_detection_active = True
    return "Pose detection started successfully."

# Route to stop pose detection
@app.route('/stop_pose_detection')
def stop_pose_detection():
    global pose_detection_active
    pose_detection_active = False
    return "Pose detection stopped successfully."

# Function to detect pose
def detect_pose(frame, pose_model):
    # Convert the BGR image to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and get the pose landmarks.
    results = pose_model.process(rgb_frame)
    
    # Check if landmarks are available.
    if results.pose_landmarks:
        # Draw the landmarks on the frame.
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    
    return frame

# Route to serve video stream with pose detection
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
