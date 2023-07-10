from flask import Flask, render_template, Response
import cv2
import numpy as np
import concurrent.futures
import threading
import time
import json

app = Flask(__name__)

# Load the pre-trained model
model_weights = 'MobileNetSSD_deploy.caffemodel'
model_config = 'MobileNetSSD_deploy.prototxt'
net = cv2.dnn.readNetFromCaffe(model_config, model_weights)

# Define the camera URLs or indices for the gates
in_gate_urls = [
    'vid4.mp4',  # Camera for 'In' Gate 1 - Modify the index or provide the URL for the specific camera
    1,  # Camera for 'In' Gate 2 - Modify the index or provide the URL for the specific camera
    2,   # Camera for 'In' Gate 3 - Modify the index or provide the URL for the specific camera
    
]

out_gate_urls = [
    'vid3.mp4',  # Camera for 'Out' Gate 1 - Modify the index or provide the URL for the specific camera
    4,  # Camera for 'Out' Gate 2 - Modify the index or provide the URL for the specific camera
    5,  # Camera for 'Out' Gate 3 - Modify the index or provide the URL for the specific camera
    
]

# Define the number of gates
num_gates = len(in_gate_urls)

# Initialize counters for each gate
in_gate_counters = [0] * num_gates
out_gate_counters = [0] * num_gates

# Lock for thread safety
counters_lock = threading.Lock()

# Function to process frames from a gate camera
def process_gate_camera(gate_index, gate_type):
    global in_gate_counters, out_gate_counters

    if gate_type == 'in':
        gate_urls = in_gate_urls
        gate_counters = in_gate_counters
    elif gate_type == 'out':
        gate_urls = out_gate_urls
        gate_counters = out_gate_counters
    else:
        return

    # Open the video capture object for the gate camera
    cap = cv2.VideoCapture(gate_urls[gate_index])

    while True:
        # Read the current frame
        ret, frame = cap.read()

        # Preprocess the frame
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # Set the input to the network
        net.setInput(blob)

        # Perform object detection
        detections = net.forward()

        # Initialize a counter for detected humans
        human_count = 0

        # Iterate over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])

                # Check if the detected object is a person (class ID 15)
                if class_id == 15:
                    # Get the bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Draw the bounding box and label on the frame
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = f"Person: {confidence * 100:.2f}%"
                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Increment the human count
                    human_count += 1

        # Update the counter for the current gate
        with counters_lock:
            gate_counters[gate_index] = human_count

        # Display the frame with gate-specific count
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    # Release the video capture
    cap.release()


@app.route('/')
def index():
    counts = [
        {'in_count': in_gate_counters[i], 'out_count': out_gate_counters[i], 'total_count': in_gate_counters[i] - out_gate_counters[i]}
        for i in range(num_gates)
    ]
    return render_template('index.html', counts=counts, num_gates=num_gates)


def count_update():
    global in_gate_counters, out_gate_counters

    while True:
        time.sleep(1)  # Update the counts every second

        with counters_lock:
            # Calculate the total count for each gate
            total_counts = [
                {'in_count': in_gate_counters[i], 'out_count': out_gate_counters[i], 'total_count': in_gate_counters[i] - out_gate_counters[i]}
                for i in range(num_gates)
            ]

        # Broadcast the counts to connected clients
        for client in clients:
            client.put(total_counts)


class Client:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()

    def put(self, count_data):
        with self.lock:
            self.queue = count_data

    def event_stream(self):
        while True:
            # Wait until there is a count update
            with self.lock:
                if not self.queue:
                    continue

                # Send the count update to the client
                count_data = self.queue
                self.queue = []

                yield f"data: {json.dumps(count_data)}\n\n"


# Maintain a list of connected clients
clients = []


@app.route('/video_feed/<gate_type>/<gate_index>')
def video_feed(gate_type, gate_index):
    if gate_type == 'in':
        gate_urls = in_gate_urls
    elif gate_type == 'out':
        gate_urls = out_gate_urls
    else:
        return

    gate_url = gate_urls[int(gate_index)]
    return Response(process_gate_camera(int(gate_index), gate_type), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/count_stream')
def count_stream():
    client = Client()
    clients.append(client)

    return Response(client.event_stream(), mimetype='text/event-stream')



if __name__ == '__main__':
    # Start the count update thread
    count_thread = threading.Thread(target=count_update, daemon=True)
    count_thread.start()

    app.run(debug=True)
