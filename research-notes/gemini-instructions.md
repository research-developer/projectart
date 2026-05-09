Architecture and Implementation Blueprint for an Interactive Projector-Based Spatial Drawing SystemThe intersection of spatial computing, computer vision, and interactive media presents a unique opportunity to transform physical environments into dynamic digital canvases. The objective of this comprehensive analysis is to architect a high-performance, low-latency interactive wall-drawing application. The system must leverage existing hardware—a projector, streaming webcams, and a highly optimized YOLO (You Only Look Once) object recognition model—to facilitate real-time gesture recognition and spatial drawing capabilities.This report provides an exhaustive technical evaluation of the hardware constraints, computer vision pipelines, latency mitigation strategies, and frontend rendering architectures required to manifest this system. It culminates in a comprehensive Product Requirements Document (PRD) designed for immediate handoff to an autonomous coding agent, ensuring a rapid and robust development cycle.Hardware Feasibility and Spatial Tracking ConstraintsThe physical architecture of a projection-mapping interactive system relies heavily on the capabilities and limitations of the input hardware. The initial consideration involves evaluating the viability of utilizing Oculus Quest (Meta Quest) controllers as the primary input devices versus bare-hand gesture recognition via a standard webcam.The Constellation Tracking Paradox and Oculus HardwareOculus Touch controllers utilize a sophisticated inside-out tracking paradigm known as Constellation tracking. The controllers are embedded with a specific pattern of infrared (IR) LEDs positioned across their surface. In a standard virtual reality setup, globally shuttered tracking cameras mounted on the headset capture these IR pulses at extremely high frame rates and short exposure times. The headset's internal processor then executes a complex pipeline: extracting bright blobs, matching them against a known 3D CAD model of the controller (LED matching), and fusing this optical data with high-frequency Inertial Measurement Unit (IMU) data to calculate a precise 6 Degrees of Freedom (6DoF) pose. Newer iterations, such as the Quest 3 Touch Plus controllers, have removed the tracking rings entirely, relying on LEDs embedded on the face of the controller and continuously fusing this data with controller-free hand tracking utilizing the headset's depth sensor.Attempting to track these controllers without the headset using standard streaming webcams introduces insurmountable technical barriers. First, standard webcams utilize rolling shutters and automatic exposure algorithms designed for the visible light spectrum. They are generally equipped with IR-cut filters. Even if the IR filter is removed, the rolling shutter introduces temporal distortion (the jello effect) that destroys the geometric relationship between the LEDs, making 3D pose estimation highly inaccurate. Second, the 6DoF tracking is inextricably linked to the headset's processing unit. Without the headset active and worn, the controllers revert to a dormant state or, at best, output 3DoF (rotational only) IMU data, which is insufficient for spatial drawing.While a subset of the development community has engineered workarounds to utilize Oculus controllers as standalone input devices, these methods are highly brittle. Techniques involving OpenVR-SpaceCalibrator, ODTKRA (a tool to keep the Oculus runtime awake), and ALVR allow for the synchronization of Oculus controllers with external headsets or PC environments. However, these methods strictly require the original Quest headset to be powered on, acting as a relay. Developers must physically tape over the proximity sensor of the headset to trick the system into an active state while keeping the headset positioned in a way that its cameras can observe the controllers. This "Rube Goldberg" configuration is entirely unsuitable for a seamless, child-friendly interactive installation in a living room. Furthermore, while there are macOS applications like Immersed and Meta Quest Remote Desktop that allow the headset to view virtual monitors, these are designed for desktop streaming into VR, not for extracting raw spatial controller data out to a macOS application. Consequently, utilizing Oculus controllers for this specific standalone projector-webcam application is strongly discouraged.The Optimal Input Paradigm: Computer Vision and Gesture RecognitionGiven the presence of a highly optimized, high-speed YOLO model, the most robust and elegant solution is to abandon hardware controllers entirely in favor of touchless hand tracking and gesture recognition. Modern neural network architectures are highly capable of segmenting human hands against complex backgrounds, even under the shifting illumination of a projected image.If the projector's light washes out bare hands—a common issue in projection mapping that degrades the YOLO model's confidence scores—a fallback mechanism can be seamlessly integrated. This involves having the user hold a distinctly colored object, a custom LED pen, or a simple retro-reflective marker. This approach guarantees near-zero hardware cost while maintaining the interactive element of the application. An LED pen, for instance, provides a high-contrast target that even a lower-resolution streaming webcam can track with sub-pixel accuracy using basic OpenCV contour detection, bypassing the need for heavy neural networks if performance becomes a bottleneck. However, assuming the YOLO model is as fast as described, full bare-hand tracking remains the primary architectural target.Evaluation of Pre-Existing Software SolutionsA critical requirement of the system design is the assessment of pre-existing tools to accelerate the development timeline. The market for projection mapping and interactive whiteboards is mature, but the intersection of the two—specifically tailored for open-source computer vision input on macOS—is a highly niche domain.Projection Mapping and Video SoftwareTraditional projection mapping software focuses heavily on geometry correction and video playback across complex, non-planar surfaces.Software ToolPrimary FunctionPlatform ConstraintsRelevance to ProjectSplashOpen-source video mapping for multi-projector domes.Linux, OSX.Low. Focuses on 3D UV unwrapping and edge blending, lacking native drawing or CV input mechanisms.MapMapFree, open-source projection mapping for artists.Windows, OSX, Linux.Low. Designed to map static videos and images to polygons (quads, triangles). It does not support real-time coordinate plotting.Digital Pressworks MapperFree software for simple surface mapping.Windows, MacOS.Low. Extremely basic application designed only to draw masks and polygons over surfaces to fit pre-rendered content.While these tools excel at their intended purposes, they are fundamentally designed for playback, not creation. They lack the real-time drawing canvas and the API endpoints necessary to ingest external $(x, y)$ coordinates generated by a Python computer vision script.Interactive Whiteboard and Smart Canvas PlatformsThe secondary category encompasses interactive whiteboards. Platforms like Canva, Miro, AFFiNE, and Mural have dominated the collaborative workspace market. Canva offers an infinite online whiteboard with real-time collaboration , while AFFiNE provides open-source structured workspaces. Educational tools like ShowMe allow for voice-over whiteboard tutorials on iPads , and OpenBoard provides an open-source cross-platform application for schools.The critical failing of these platforms for this specific use case is their input paradigm. They are rigidly coupled to standard Human Interface Device (HID) events—namely, mouse clicks, stylus pressure, or capacitive touch. While it is theoretically possible to write a global operating system script that intercepts the YOLO model's coordinates and translates them into simulated macOS mouse events (e.g., using Python's pyautogui or pynput) to draw inside Canva or OpenBoard, this introduces severe user experience degradation. Simulated mouse clicks often interfere with background OS tasks, lack support for nuanced brush pressure, and cannot easily interpret complex gestures (like pinching to open a specific color menu) without triggering unintended system-wide macros.Specialized Computer Vision ToolkitsThere are a few open-source projects that attempt to bridge computer vision and projection directly.InteractiveProjectionLib: This is an open-source C++ library based on OpenCV designed specifically for interactive projection applications. It includes methods for capturing chessboard corners to find homography and recursively training background subtractors. However, it is built on the outdated OpenCV 3.0 standard and requires complex CMake compilation , making it cumbersome to integrate with modern Python-based YOLO pipelines.PapARt: A Software Development Kit (SDK) created by Inria for interactive projection mapping, built on top of Processing, OpenCV, and JavaCV. It supports marker tracking and depth-camera finger tracking. While powerful, its reliance on the Java ecosystem introduces friction when attempting to integrate a custom Python YOLO model, which typically demands a seamless Python environment or complex inter-process communication bridging Java and Python.The Superior Approach: A Custom Decoupled ArchitectureThe exhaustive analysis of pre-existing tools reveals that no single off-the-shelf application perfectly accommodates a custom, high-speed YOLO model paired with a streaming webcam for spatial drawing. Attempting to shoehorn computer vision data into a generic tool like Canva via simulated mouse clicks is fragile. Conversely, wrestling with outdated C++ libraries like InteractiveProjectionLib slows down development.Therefore, the fastest, most resilient, and most extensible path forward is to build a bespoke application using a decoupled microservices architecture. The system will comprise a Python backend dedicated solely to video ingestion, YOLO inference, and spatial mathematics. This backend will communicate via real-time WebSockets to a lightweight, browser-based HTML5 frontend. This approach leverages the user's existing YOLO asset, requires no compilation of C++ libraries, and allows for infinite customization of the drawing interface.Latency Mitigation in Networked Video StreamsThe hardware architecture relies on streaming webcams, which inherently introduce latency compared to directly attached USB (UVC) cameras. Latency in an interactive drawing application is catastrophic for the user experience; if the projected brush stroke trails the user's physical hand by more than 50 to 100 milliseconds, the illusion of direct physical interaction completely breaks down.Protocol and Decoding OptimizationStandard IP cameras and streaming webcams typically utilize the Real-Time Streaming Protocol (RTSP) encoded with H.264 or H.265 compression algorithms. The primary sources of latency in this pipeline are the network buffer and the decoding overhead. Applications utilizing OpenCV's VideoCapture function often rely on the FFmpeg backend, which, by default, buffers several frames to ensure smooth, jitter-free video playback. This buffering creates an unacceptable delay for interactive computer vision.To eradicate this delay, the ingestion pipeline must be aggressively optimized at the transport layer. When configuring the VideoCapture object in Python, environment variables or FFmpeg flags must be explicitly passed to disable this buffering behavior. Flags such as -fflags nobuffer, -flags low_delay, and -tune zerolatency force the decoder to process frames immediately upon arrival, bypassing the standard B-frame (bi-directional predictive frame) buffering.Alternative Transport Architectures: ZeroMQ and MJPEGIf the streaming webcams can be configured at the source, abandoning highly compressed H.264 streams in favor of Motion JPEG (MJPEG) over a lightweight messaging protocol like ZeroMQ (ZMQ) drastically reduces latency. MJPEG sends each frame as a complete, independent JPEG image, eliminating the inter-frame dependencies that cause decoding delays in modern video codecs.By utilizing a publisher-subscriber (PUB/SUB) architecture in ZeroMQ, the Python backend can receive encoded JPEG strings in real-time. A critical configuration in this setup is manipulating the High Water Mark (SNDHWM and RCVHWM). By setting this value to a minimum (e.g., 1 or 10), the socket will automatically drop older frames if the network or the receiving thread falls behind, ensuring that the computer vision pipeline is only ever analyzing the absolute most recent image captured by the camera.Thread Isolation StrategiesRegardless of the transport protocol (RTSP or ZMQ), the video capture loop must be completely decoupled from the YOLO inference loop. A dedicated background daemon thread must continuously read from the camera stream. This thread's sole responsibility is to place the newest frame into a thread-safe Queue of size 1, aggressively discarding older, unprocessed frames. The primary application thread, which houses the YOLO inference engine, then simply pulls the most recent frame from this Queue. This architecture guarantees that heavy neural network processing never causes the video feed to back up, effectively neutralizing the latency bottleneck inherent in cv2.read() operations.The Computer Vision Pipeline: Spatial Mathematics and CalibrationTo allow a user to draw seamlessly on a physical wall, the system must continuously translate the 2D coordinates of the user's hand, as observed by the streaming webcam, into the 2D coordinate space of the projector. Because the webcam and the projector are not physically co-located and possess different optical properties (field of view, lens distortion), their optical axes differ, resulting in severe perspective distortion.Planar Homography and Perspective TransformationThe mathematical solution to mapping points between two disparate perspective views of a flat surface is a planar homography. Because the drawing surface (the wall) is a flat 2D plane, the mapping between the camera's image plane and the projector's image plane can be precisely described by a $3 \times 3$ transformation matrix, $H$.The relationship between a point in the camera view, denoted as $(u, v)$, and the corresponding point in the projector space, denoted as $(x, y)$, is defined using homogeneous coordinates. This is expressed in the following equation:$$s \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$Here, $s$ represents an arbitrary scale factor. To compute this homography matrix, the system requires a rigid calibration phase. The standard and most mathematically sound approach involves projecting a known geometric pattern—typically a high-contrast chessboard or a sequence of structured light Gray codes—onto the wall using the projector. The streaming webcam captures these projected patterns, and computer vision algorithms extract the precise sub-pixel coordinates of the corner points.By obtaining a minimum of four corresponding point pairs—though substantially more are used in practice to minimize optical error via Random Sample Consensus (RANSAC)—OpenCV's cv2.findHomography() function calculates the optimal matrix $H$. Once $H$ is computed and stored in memory, any hand coordinate $(u, v)$ detected by the YOLO model in real-time can be multiplied by $H$ using cv2.perspectiveTransform() to find the exact pixel $(x, y)$ where the projector should render the digital brush stroke.Calibration PhaseSystem MechanismObjective1. Pattern GenerationProject a highly defined, high-contrast $9 \times 6$ chessboard image onto the wall.Provides easily identifiable geometric intersections.2. Image CaptureWebcam captures the projected image on the physical wall.Acquires the distorted camera-view coordinate data.3. Feature ExtractionUtilize cv2.findChessboardCorners() on the captured frame.Locates the sub-pixel coordinates of the intersections $(u, v)$.4. Matrix ComputationExecute cv2.findHomography(camera_pts, projector_pts).Generates the foundational $3 \times 3$ perspective transformation matrix.5. Real-time MappingExecute cv2.perspectiveTransform() continuously.Converts dynamic YOLO hand bounding box centers directly to canvas coordinates.For a more child-friendly and immediate setup, an interactive manual calibration routine can be implemented. The projector displays four distinct target circles at the corners of its display area. The user holds their hand (or the colored marker) over each target sequentially. The YOLO model detects the hand at these four extremes, logging the $(u, v)$ coordinates. These four points are then mapped directly to the known resolution boundaries of the projector (e.g., (0,0), (1920,0), (1920,1080), (0,1080)) to generate the homography matrix instantly.Advanced Gesture Recognition: YOLO and MediaPipe IntegrationThe core intelligence of the system lies in translating raw video frames into actionable drawing commands. The integration of the user's custom, high-speed YOLO model serves as the foundational detection layer, but it must be augmented to provide the nuanced control required for a compelling drawing application.Open-Vocabulary Detection and Region of Interest ExtractionModern YOLO iterations, such as YOLOv8 or open-vocabulary variants like YOLOE (Real-Time Seeing Anything), excel at high-speed bounding box generation and object localization. These architectures utilize convolutional backbones for feature extraction and multi-scale fusion to achieve real-time detection. In this specific system architecture, the YOLO model is tasked strictly with localizing the user's hand within the wide-angle frame and cropping the Region of Interest (ROI).By restricting all subsequent deep learning analysis to a small bounding box rather than the full 1080p or 4K camera frame, the processing pipeline executes exponentially faster. This is critical for maintaining high frame rates on macOS hardware.Semantic Gesture Classification via Skeletal KinematicsTo allow the user to change brushes, alter stroke width, or switch colors without touching a physical keyboard, the system must classify specific hand poses. While a YOLO model can theoretically be trained to recognize distinct classes (e.g., hand_pointing, hand_open, hand_fist), relying solely on bounding box classifications for precise drawing coordinates is fundamentally flawed. Bounding boxes inherently suffer from spatial jitter; as the hand moves, the boundaries of the box fluctuate, causing the center point to tremble wildly. This translates to jagged, erratic lines on the projector canvas.A significantly superior architectural pattern involves a two-stage pipeline:Stage 1: YOLO ROI Extraction. The fast YOLO model identifies the hand and outputs a tight bounding box, eliminating background noise.Stage 2: Landmark Extraction. Google's MediaPipe Hand Tracking framework processes only the YOLO-extracted ROI. MediaPipe outputs a highly precise, 21-point 3D skeletal map of the hand, providing exact coordinates for fingertips, knuckles, and the palm center.By analyzing the Euclidean distances and geometric angles between specific skeletal nodes (e.g., the tip of the index finger versus the tip of the thumb), the Python backend can programmatically define highly robust state machines for complex gestures. This approach is vastly more stable than bounding box classification.Semantic GestureSkeletal Logic (MediaPipe Nodes)Application ActionDraw ModeNode 8 (Index Tip) y-axis < Node 6; Other fingers folded into the palm.Apply Ink. Coordinates of Node 8 are mapped to the canvas via Homography.Pinch GestureDistance between Node 8 (Index) and Node 4 (Thumb) < Defined Threshold.Menu Interaction. Triggers a UI overlay or cycles through a color palette.Open PalmNodes 8, 12, 16, 20 fully extended radially from the palm center (Node 0).Erase. Clears the stroke array or acts as a broad eraser tool.FistAll distal phalanges nodes located below the proximal phalanges nodes.Hover Mode. Halts input, allowing the user to move their hand without drawing.Temporal Smoothing AlgorithmsEven with the precision of MediaPipe landmarks, raw coordinate data extracted from optical video frames contains high-frequency noise. If these raw coordinates are passed directly to the rendering canvas, the resulting lines will still appear slightly erratic. The Python backend must apply a temporal smoothing filter to the $(u, v)$ coordinates of the index fingertip before calculating the perspective transformation. Implementing a One Euro Filter or a simple Exponential Moving Average (EMA) algorithm effectively filters out this optical noise while preserving the rapid, deliberate movements of the user's hand, ensuring the digital ink flows naturally and mimics the physics of a physical brush.Application Layer: Rendering and Asynchronous CommunicationThe visual output—what the user actually sees projected onto the physical wall—must be rendered by a lightweight, responsive client. The architectural requirements allow for a web browser, an Electron application, or a native macOS app. Given the explicit desire for rapid development, pre-existing tool availability, and cross-platform flexibility, a web-based frontend utilizing HTML5 Canvas is the optimal path forward.The Rendering Engine: p5.js and p5.brush.jsTo ensure the application is genuinely engaging and magical for a child, rendering sterile, 1-pixel vector lines is insufficient. The application must visually simulate organic, physical media. The p5.js library, specifically tailored for creative coding, generative art, and accessible interactive design, serves as the ideal rendering engine.Furthermore, extending the core p5.js environment with the open-source p5.brush.js library unlocks complex vector fields, custom hatching, and highly realistic natural fill effects. This allows the gesture recognition system to control dynamic brush behaviors seamlessly. For instance, the velocity of the hand movement (calculated by the Python backend across consecutive frames) can be transmitted to the frontend to dynamically alter the stroke thickness and texture opacity in the browser, creating a deeply tactile and responsive visual experience.Real-Time Telemetry via WebSocketsThe architecture is explicitly decoupled to maximize performance: a Python backend handles the heavy lifting of video decoding, YOLO inference, MediaPipe skeletal extraction, and matrix mathematics, while a browser-based frontend handles the visual rendering pipeline.To bridge these two distinct environments with sub-millisecond latency, standard HTTP REST requests are entirely obsolete. The system must rely on persistent, bi-directional WebSocket connections. Implementing Socket.IO or Python's native websockets library provides the necessary high-throughput transport layer.In this architecture, the Python backend acts as the socket server. At every processed frame (ideally scaling to 30 to 60 times per second), the backend computes the transformed $(x, y)$ projector coordinate and the current semantic gesture state. It serializes this data into a minimal JSON payload and emits it over the socket. The p5.js frontend client listens continuously for these events and immediately calls its line() or custom p5.brush.js rendering functions upon receipt, ensuring the visual feedback remains tightly synchronized with the physical movement.Product Requirements Document (PRD)The following PRD is formatted for immediate ingestion by an autonomous coding agent (such as Claude) to rapidly bootstrap the codebase based on the strict architectural constraints established in the preceding analysis.PRD: Interactive Spatial Canvas (Projector-Based Drawing App)1. Product VisionName: ProjectorCanvas (Internal Code Name)Purpose: An interactive, touchless drawing application that allows users to draw dynamically on a physical wall using a projector, a streaming webcam, and advanced hand gesture recognition.Target Audience: Children and creative hobbyists. The emphasis is entirely on immediate responsiveness, organic brush textures, and magical "touchless" interactions that mimic physical reality.Technology Stack:Backend Server: Python 3.10+, OpenCV, YOLO (Ultralytics custom model), MediaPipe Hands, Flask-SocketIO, NumPy.Frontend Client: HTML5, CSS3, JavaScript (ES6), p5.js, p5.brush.js, Socket.IO client.2. System Architecture SpecificationsThe system is divided into two distinct, loosely coupled microservices communicating asynchronously over localhost WebSockets.2.1 Backend Pipeline (Python)Video Ingestion Engine: Must connect to the streaming webcam using explicit zero-latency OpenCV flags (cv2.CAP_FFMPEG, nobuffer). Must execute in a dedicated daemon thread to completely prevent frame buffering, maintaining a Queue size of 1.Vision Inference Engine:Executes the provided custom YOLO model inference to locate the hand within the wide-angle frame.Passes the cropped ROI to MediaPipe Hands to extract 21 skeletal landmarks, ensuring high-speed processing.Determines the current semantic gesture state (Draw, Hover, Erase, Color Change) based on rigid kinematic geometry logic.Applies an Exponential Moving Average (EMA) smoothing algorithm to the raw coordinates of the index fingertip to eradicate optical jitter.Spatial Mathematics Module:Provides a manual calibration routine. When triggered, the frontend projects a UI with 4 distinct corner targets. The user clicks these targets on the physical wall (holding an object), or uses the mouse on the raw webcam feed window to select the 4 corners of the projected area.Computes and saves the $3 \times 3$ Homography matrix to disk.Transforms webcam coordinates $(u, v)$ to projector/browser viewport coordinates $(x, y)$ in real-time.Telemetry Server: Emits a serialized JSON payload at a target rate of 30-60Hz to the frontend client.2.2 Frontend Application (Web Browser / p5.js)Canvas Engine: Runs a full-screen, borderless p5.js canvas on the display output routed to the projector.Brush Logic: Utilizes p5.brush.js to render organic, textured strokes rather than basic geometric lines. Stroke weight must scale dynamically based on the velocity parameter received from the backend.State Management: Reacts instantly to WebSocket payloads to toggle the visibility of UI elements (e.g., rendering a floating color palette when a PINCH gesture state is received, or applying a fade-out mask when ERASE is received).3. Data Contracts and API DefinitionsWebSocket Event: gesture_dataDirection: Backend $\rightarrow$ FrontendPayload Structure:JSON{
  "timestamp": 1715201045.123,
  "state": "DRAW", 
  "x": 1450.5,
  "y": 800.2,
  "velocity": 12.4,
  "color_hex": "#FF5733"
}
Valid States:HOVER: Cursor indicator is visible on the wall, no ink is applied.DRAW: Apply ink between the previous coordinate and the current coordinate.ERASE: Trigger a canvas wipe or activate a broad eraser brush.MENU: Halt drawing, open radial color picker interface at the current $(x, y)$ location.4. End-User WorkflowsInitialization: The user boots the Python server and opens index.html in a full-screen browser window on the projector display.Calibration: If no saved matrix exists, the system enters calibration mode. Four targets appear. The user points to each target, and the system locks the spatial boundaries.Interaction: The user raises their index finger to draw. Pinching the thumb and index finger pulls up a color wheel. Moving the pinched fingers selects a color. Releasing the pinch commits the color and returns to HOVER mode. Opening the hand flat clears the canvas.5. Code Implementation Stubs for Autonomous HandoffThe following code stubs establish the rigid architectural scaffolding required for the coding agent to begin implementation immediately.5.1 Python Backend Server (app.py)Pythonimport cv2
import numpy as np
import threading
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
from ultralytics import YOLO
import mediapipe as mp

app = Flask(__name__, template_folder="static", static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ---------------------------------------------------------
# Configuration & Globals
# ---------------------------------------------------------
# Webcams that stream require strict latency optimizations
WEBCAM_URL = "rtsp://username:password@ip_address/stream" 
H_MATRIX = np.eye(3) # Homography matrix placeholder
LATEST_FRAME = None

# Initialize Models
# The user provides a modified YOLO object recognition model
model_yolo = YOLO('custom_yolo_model.pt') 
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ---------------------------------------------------------
# Zero-Latency Frame Capture Thread
# ---------------------------------------------------------
def capture_frames():
    global LATEST_FRAME
    # Explicit FFmpeg flags to eradicate network buffering
    cap = cv2.VideoCapture(WEBCAM_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            LATEST_FRAME = frame
        else:
            time.sleep(0.01)

# ---------------------------------------------------------
# Vision Processing & Telemetry Thread
# ---------------------------------------------------------
def process_vision():
    global LATEST_FRAME, H_MATRIX
    
    # Initialize variables for velocity calculation and EMA smoothing
    prev_cx, prev_cy = 0, 0
    alpha_ema = 0.6 
    
    while True:
        if LATEST_FRAME is None:
            time.sleep(0.01)
            continue
            
        frame = LATEST_FRAME.copy()
        
        # 1. Execute YOLO inference to locate the hand bounding box
        results = model_yolo(frame, classes=, verbose=False) 
        
        # 2. Extract ROI and pass to MediaPipe for skeletal kinematics
        # STUB: Add ROI cropping logic here based on YOLO bbox coordinates
        mp_results = hands_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if mp_results.multi_hand_landmarks:
            for hand_landmarks in mp_results.multi_hand_landmarks:
                # Extract Index Finger Tip (Node 8)
                h, w, c = frame.shape
                raw_cx = int(hand_landmarks.landmark.x * w)
                raw_cy = int(hand_landmarks.landmark.y * h)
                
                # Apply Exponential Moving Average (EMA) for jitter reduction
                cx = (alpha_ema * raw_cx) + ((1 - alpha_ema) * prev_cx)
                cy = (alpha_ema * raw_cy) + ((1 - alpha_ema) * prev_cy)
                
                # Calculate velocity magnitude
                velocity = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                prev_cx, prev_cy = cx, cy
                
                # 3. Apply Planar Homography Transformation
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                transformed_pt = cv2.perspectiveTransform(pt, H_MATRIX)
                proj_x, proj_y = transformed_pt
                
                # 4. Semantic Gesture Logic 
                # STUB: Implement geometric checks (e.g., if Node 8 y < Node 6 y -> DRAW)
                state = "DRAW" 
                
                # 5. Transmit Payload via WebSockets
                payload = {
                    "state": state,
                    "x": float(proj_x),
                    "y": float(proj_y),
                    "velocity": float(velocity),
                    "color_hex": "#FF5733"
                }
                socketio.emit('gesture_data', payload)
        
        # Maintain process loop rate
        time.sleep(0.016) 

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Initialize background threads for isolated processing
    threading.Thread(target=capture_frames, daemon=True).start()
    threading.Thread(target=process_vision, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
5.2 Frontend Web Canvas (static/index.html & static/sketch.js)index.htmlHTML<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Projector Canvas - Interactive Media Environment</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.js"></script>
    <script src="https://cdn.jsdelivr.net/gh/acamposq/p5.brush.js/p5.brush.min.js"></script>
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
    <style>
        body { margin: 0; padding: 0; overflow: hidden; background-color: #000; cursor: none; }
        canvas { display: block; }
    </style>
</head>
<body>
    <script src="sketch.js"></script>
</body>
</html>
sketch.jsJavaScriptlet socket;
let currentX = 0, currentY = 0;
let prevX = 0, prevY = 0;
let gestureState = "HOVER";
let brushColor = "#FFFFFF";

function setup() {
    createCanvas(windowWidth, windowHeight);
    background(0); // Initialize pitch black background for projection mapping
    
    // Initialize p5.brush.js texturing engine
    brush.scale(1.5);
    brush.add("watercolor", {
        type: "custom",
        weight: 5,
        vibration: 0.1,
        opacity: 0.8
    });
    brush.set("watercolor", brushColor, 5);

    // Initialize WebSockets connection to Python telemetry server
    socket = io.connect('http://localhost:5000');
    
    // Register asynchronous event listener for coordinate payloads
    socket.on('gesture_data', function(data) {
        prevX = currentX;
        prevY = currentY;
        currentX = data.x;
        currentY = data.y;
        gestureState = data.state;
        
        // Dynamically scale brush weight based on user's hand velocity
        let weight = map(data.velocity, 0, 50, 2, 20);
        weight = constrain(weight, 2, 20);
        brush.set("watercolor", data.color_hex, weight);
    });
}

function draw() {
    // Process ERASE state: Apply high-alpha mask to clear canvas
    if (gestureState === "ERASE") {
        background(0, 50); 
        return;
    }
    
    // Process MENU state: Render radial UI overlay
    if (gestureState === "MENU") {
        push();
        noStroke();
        fill(200, 200, 200, 150);
        circle(currentX, currentY, 100); // STUB: Expand to full radial color picker
        pop();
        return;
    }
    
    // Process HOVER state: Render tracking reticle without applying ink
    if (gestureState === "HOVER") {
        push();
        noStroke();
        fill(255, 150);
        circle(currentX, currentY, 8);
        pop();
    }
    
    // Process DRAW state: Execute textured brush stroke
    if (gestureState === "DRAW") {
        brush.line(prevX, prevY, currentX, currentY);
    }
}

// Ensure rendering canvas strictly adheres to dynamic projector resolution bounds
function windowResized() {
    resizeCanvas(windowWidth, windowHeight);
}
6. Strategic Directives for the Autonomous Coding AgentUpon handing this architectural framework and PRD to the coding assistant, the agent must be directed to execute the following iterative development phases sequentially, ensuring stability at each tier before proceeding.Phase 1: Environment Stabilization and Calibration. The agent's immediate priority must be the implementation of the planar homography calibration script. This should manifest as a standalone Python utility. It must project a 4-point bounding quad onto the display output, capture the incoming webcam frame, and prompt the user to manually select the 4 corresponding projection boundaries in the raw webcam feed using OpenCV's setMouseCallback. It will then calculate and serialize the H_MATRIX to a persistent .npy file, establishing the foundational spatial coordinate map.Phase 2: Vision Pipeline Optimization. The agent must adapt the provided Python backend stub to instantiate the specific custom YOLO model provided by the user. The bounding box outputs derived from YOLO inference must be utilized strictly to calculate cropping coordinates. The agent must ensure that MediaPipe operates exclusively within this cropped tensor to minimize computational overhead and maximize throughput.Phase 3: Mathematical Smoothing and Kinematics. The agent must design and implement an object-oriented Exponential Moving Average (EMA) class within the Python backend. This class will maintain state across frames to dampen the high-frequency jitter native to raw optical tracking. Following this, the agent will encode the geometric rulesets for the semantic gesture state machine (DRAW, PINCH, PALM) based on the relative Cartesian distances between MediaPipe nodes.Phase 4: Frontend Aesthetics and Interaction Design. Finally, the agent must enrich the p5.js frontend. This involves expanding the "MENU" state stub into a functional, interactable radial color wheel that the user can navigate via the "PINCH" gesture. The agent must utilize the native capabilities of p5.js and p5.brush.js to ensure the visual feedback—such as the transition from a hovering reticle to a flowing watercolor stroke—is instantaneous, deeply immersive, and highly responsive to the physical dynamics of the user's movements.