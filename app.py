import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import time
import os
from PIL import Image
import tempfile
import io
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


class YOLOv8Detector:
    def __init__(self, model_path, labels, confidence_threshold=0.45, iou_threshold=0.45, input_width=640, input_height=640):
        """
        Initialize the YOLOv8 detector with ONNX runtime
        
        Args:
            model_path: Path to the YOLOv8 ONNX model
            labels: List of class labels
            confidence_threshold: Minimum confidence for a box to be detected
            iou_threshold: IOU threshold for NMS
            input_width: Width of input image to the model
            input_height: Height of input image to the model
        """
        self.labels = labels
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_width = input_width
        self.input_height = input_height
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        print(f"Loading ONNX model from {model_path}")
        
        # Initialize ONNX runtime session - just use CPU to avoid CUDA issues
        providers = ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"Using providers: {self.session.get_providers()}")
        except Exception as e:
            print(f"Failed to initialize ONNX session: {e}")
            raise
        
        # Get model info
        self.get_input_details()
        self.get_output_details()
        print(f"Model loaded successfully. Input shape: {self.input_shape}")
    
    def get_input_details(self):
        """Get model input details"""
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.channels = 3  # Usually fixed at 3 for RGB
        
    def get_output_details(self):
        """Get model output details"""
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
    
    def prepare_input(self, image):
        """
        Prepare image for inference
        """
        self.img_height, self.img_width = image.shape[:2]
        
        # Resize input image
        input_img = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert the image color to RGB (yolo expects RGB images)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # Scale input pixel values to 0 to 1
        input_img = input_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :]
        
        return input_tensor
    
    def process_output(self, outputs):
        """
        Process ONNX model output for YOLOv8
        """
        output = outputs[0]
        
        # Handle different possible output formats
        if len(output.shape) == 3:  # [batch, boxes, elements]
            predictions = np.squeeze(output)  # Remove batch dimension
            
            # YOLOv8 format
            if predictions.shape[0] == 84 or predictions.shape[0] > 84:
                # Transpose to have predictions in expected format [num_boxes, elements]
                predictions = predictions.transpose(1, 0)
            
            # Extract boxes, scores, and class IDs
            boxes = predictions[:, :4]  # First 4 elements are x, y, w, h
            scores = predictions[:, 4:].max(axis=1)  # Get max confidence across classes
            class_ids = predictions[:, 4:].argmax(axis=1)  # Get class with max confidence
            
            # Filter by confidence threshold
            mask = scores >= self.confidence_threshold
            boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
            
            # If no detections, return empty lists
            if len(boxes) == 0:
                return [], [], []
            
            # Convert boxes to xyxy format and scale to original image size
            boxes = self.xywh2xyxy(boxes)
            boxes = self.rescale_boxes(boxes)
            
            # Apply NMS
            try:
                indices = cv2.dnn.NMSBoxes(
                    boxes.tolist(), scores.tolist(), self.confidence_threshold, self.iou_threshold
                )
                
                # Extract results after NMS
                if len(indices) > 0:
                    if isinstance(indices, np.ndarray):
                        if len(indices.shape) == 2:
                            indices = indices.flatten()
                        
                    final_boxes = [boxes[i] for i in indices]
                    final_scores = [scores[i] for i in indices]
                    final_class_ids = [class_ids[i] for i in indices]
                    
                    return final_boxes, final_scores, final_class_ids
                else:
                    return [], [], []
            except Exception as e:
                print(f"Error during NMS: {e}")
                return [], [], []
        
        # Fallback for other model output formats
        else:
            predictions = np.squeeze(output)
            
            # Get scores and filter by threshold
            if predictions.shape[1] > 4:
                scores = np.max(predictions[:, 4:], axis=1)
                predictions = predictions[scores > self.confidence_threshold]
                scores = scores[scores > self.confidence_threshold]
                
                if len(scores) == 0:
                    return [], [], []
                
                # Get class IDs
                class_ids = np.argmax(predictions[:, 4:], axis=1)
                
                # Extract and process boxes
                boxes = self.extract_boxes(predictions)
                
                # Apply NMS
                try:
                    indices = cv2.dnn.NMSBoxes(
                        boxes.tolist(), scores.tolist(), self.confidence_threshold, self.iou_threshold
                    )
                    
                    if len(indices) > 0:
                        if isinstance(indices, np.ndarray):
                            if len(indices.shape) == 2:
                                indices = indices.flatten()
                        
                        final_boxes = [boxes[i] for i in indices]
                        final_scores = [scores[i] for i in indices]
                        final_class_ids = [class_ids[i] for i in indices]
                        
                        return final_boxes, final_scores, final_class_ids
                    else:
                        return [], [], []
                except Exception as e:
                    print(f"Error during NMS: {e}")
                    return [], [], []
            else:
                return [], [], []
    
    def extract_boxes(self, predictions):
        """Extract boxes from predictions"""
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = self.xywh2xyxy(boxes)
        return boxes
    
    def rescale_boxes(self, boxes):
        """Rescale boxes to original image dimensions"""
        scale_x = self.img_width / self.input_width
        scale_y = self.img_height / self.input_height
        
        scaled_boxes = boxes.copy()
        scaled_boxes[:, 0] *= scale_x  # x1 or x
        scaled_boxes[:, 1] *= scale_y  # y1 or y
        scaled_boxes[:, 2] *= scale_x  # x2 or w
        scaled_boxes[:, 3] *= scale_y  # y2 or h
        
        return scaled_boxes
    
    def xywh2xyxy(self, x):
        """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    def draw_detections(self, image, boxes, scores, class_ids):
        """Draw detections on the image"""
        # Make a copy to avoid modifying the original
        img_copy = image.copy()
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            # Ensure class_id is within the valid range
            if class_id < 0 or class_id >= len(self.labels):
                label = f'Unknown ({class_id}): {score:.2f}'
            else:
                label = f'{self.labels[class_id]}: {score:.2f}'
            
            # Make sure all box coordinates are integers
            x1, y1, x2, y2 = [int(round(coord)) for coord in box]
            
            # Make sure coordinates are within image bounds
            x1 = max(0, min(x1, image.shape[1] - 1))
            y1 = max(0, min(y1, image.shape[0] - 1))
            x2 = max(0, min(x2, image.shape[1] - 1))
            y2 = max(0, min(y2, image.shape[0] - 1))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Generate a color for the class
            color_h = (class_id * 10) % 180
            color_hsv = np.array([[[color_h, 255, 255]]], dtype=np.uint8)
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            color = tuple([int(c) for c in color_bgr])
            
            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Get size of the text
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Make sure text background stays within image bounds
            if y1 - text_height - 5 < 0:
                # If label would be above the image, put it inside the bounding box
                text_y = y1 + text_height + 5
                cv2.rectangle(img_copy, (x1, y1), (x1 + text_width, y1 + text_height + 5), color, -1)
                cv2.putText(img_copy, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            else:
                # Draw filled rectangle for text background
                cv2.rectangle(img_copy, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
                # Add text
                cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img_copy
    
    def detect(self, image):
        """
        Detect objects in the input image
        """
        # Create a copy of the image for drawing
        processed_image = image.copy()
        
        # Prepare input
        input_tensor = self.prepare_input(image)
        
        # Perform inference
        start = time.time()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        end = time.time()
        
        # Process output
        boxes, scores, class_ids = self.process_output(outputs)
        
        # Draw detections
        processed_image = self.draw_detections(processed_image, boxes, scores, class_ids)
        
        # Calculate inference time
        inference_time = end - start
        fps = 1 / inference_time if inference_time > 0 else 0
        
        # Display FPS
        cv2.putText(processed_image, f'FPS: {fps:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return boxes, scores, class_ids, processed_image


class VideoProcessor(VideoProcessorBase):
    def __init__(self, detector):
        self.detector = detector
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Use your existing YOLOv8Detector
        boxes, scores, class_ids, processed_img = self.detector.detect(img)
        
        # Return processed frame
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


# COCO dataset labels
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


def load_custom_css():
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
        background: linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    
    .card {
        border-radius: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: rgba(255, 255, 255, 0.02);
    }
    
    .card:hover {
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
    }
    
    .success-message {
        background-color: rgba(34, 197, 94, 0.1);
        color: #22c55e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .info-message {
        background-color: rgba(59, 130, 246, 0.1);
        color: #3b82f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .detection-box {
        background-color: #f8fafc;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    @media (prefers-color-scheme: dark) {
        .detection-box {
            background-color: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(226, 232, 240, 0.1);
        }
    }
    </style>
    """, unsafe_allow_html=True)


def display_header():
    st.markdown("""
    <div style="text-align: center;">
        <div class="main-title">YOLO Vision Pro</div>
        <div class="subtitle">Advanced AI-Powered Object Detection</div>
    </div>
    """, unsafe_allow_html=True)


def display_detection_results(boxes, scores, class_ids):
    if len(boxes) > 0:
        st.markdown('<div class="success-message">✅ Objects detected successfully!</div>', unsafe_allow_html=True)
        
        st.markdown("<h3>Detection Results</h3>", unsafe_allow_html=True)
        
        # Display total counts by class
        class_counts = {}
        for class_id in class_ids:
            class_name = COCO_LABELS[class_id] if 0 <= class_id < len(COCO_LABELS) else f"Unknown ({class_id})"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Create columns for summary stats
        cols = st.columns(3)
        with cols[0]:
            st.metric("Total Objects", len(boxes))
        with cols[1]:
            st.metric("Object Classes", len(class_counts))
        with cols[2]:
            top_class = max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else "None"
            st.metric("Most Common", f"{top_class}")
        
        # Display results in a cleaner table
        results = []
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = [int(round(coord)) for coord in box]
            class_name = COCO_LABELS[class_id] if 0 <= class_id < len(COCO_LABELS) else f"Unknown ({class_id})"
            results.append({
                "Object": class_name,
                "Confidence": f"{score:.2f}",
                "Location": f"[{x1}, {y1}, {x2}, {y2}]"
            })
        
        st.table(results)
    else:
        st.markdown('<div class="info-message">ℹ️ No objects detected in this image.</div>', unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path, conf_thresh, iou_thresh):
    """Cache the model to prevent reloading"""
    try:
        detector = YOLOv8Detector(
            model_path=model_path,
            labels=COCO_LABELS,
            confidence_threshold=conf_thresh,
            iou_threshold=iou_thresh
        )
        return detector
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def main():
    # Set page config
    st.set_page_config(
        page_title="YOLO Vision Pro | Object Detection",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Display header
    display_header()
    
    # Sidebar configuration
    with st.sidebar:
        st.title("Model Settings")
        
        # Model selection with simplified options
        model_option = st.selectbox(
            "Select Model",
            ["YOLOv8 Nano (⚡ Fast, 4.3MB)", 
             "YOLOv8 Small (🚀 Balanced, 11.4MB)", 
             "YOLOv8 Large (🎯 Accurate, 43.7MB)"],
            index=0
        )
        
        # Map model options to file paths
        model_paths = {
            "YOLOv8 Nano (⚡ Fast, 4.3MB)": "yolov8n.onnx",
            "YOLOv8 Small (🚀 Balanced, 11.4MB)": "yolov8s.onnx",
            "YOLOv8 Large (🎯 Accurate, 43.7MB)": "yolov8l.onnx"
        }
        
        # Get selected model path
        model_path = model_paths[model_option]
        
        # Check if model exists
        if not os.path.exists(model_path):
            st.error(f"⚠️ Model file not found: {model_path}")
            st.info("📁 Please make sure to place the ONNX model files in the same directory as this script.")
            return
        
        st.markdown("---")
        st.subheader("Detection Parameters")
        
        # Detection parameters
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.3, 
            step=0.05,
            help="Minimum confidence level for detections"
        )
        
        iou_threshold = st.slider(
            "IOU Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.45, 
            step=0.05,
            help="Overlap threshold for bounding boxes"
        )
        
        st.markdown("---")
        
        # Save video option
        save_video = st.checkbox("Save Video", value=False)
    
    # Load the model (cached)
    detector = load_model(model_path, confidence_threshold, iou_threshold)
    
    if detector is None:
        st.error("Failed to load the model. Please check the model path and try again.")
        return
        
    # Create tabs for different options
    tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "🎥 Webcam", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Convert the file to an opencv image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Use columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, channels="BGR", use_container_width=True)
            
            # Process button
            detect_btn = st.button("🔍 Detect Objects", type="primary", use_container_width=True)
            
            if detect_btn:
                with st.spinner("Processing image..."):
                    try:
                        # Perform detection
                        boxes, scores, class_ids, processed_image = detector.detect(image)
                        
                        with col2:
                            st.subheader("Detection Results")
                            st.image(processed_image, channels="BGR", use_container_width=True)
                        
                        # Display detection details
                        display_detection_results(boxes, scores, class_ids)
                        
                        # Download option
                        if len(boxes) > 0:
                            processed_img = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(processed_img)
                            buf = io.BytesIO()
                            pil_img.save(buf, format="PNG")
                            
                            st.download_button(
                                label="💾 Download Processed Image",
                                data=buf.getvalue(),
                                file_name="detection_result.png",
                                mime="image/png"
                            )
                    except Exception as e:
                        st.error(f"⚠️ Error: {str(e)}")
    
    with tab2:
        st.header("Webcam Object Detection")
        
        webcam_method = st.radio(
            "Choose webcam method:",
            ["WebRTC", "OpenCV"],
            horizontal=True
        )
        
        if webcam_method == "WebRTC":
            try:
                st.info("📹 Using WebRTC for webcam access. Allow camera permissions when prompted.")
                
                # Create video processor for WebRTC
                processor = VideoProcessor(detector)
                
                # Define STUN servers for WebRTC
                rtc_configuration = {
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }
                
                # Create WebRTC streamer
                webrtc_ctx = webrtc_streamer(
                    key="object-detection",
                    video_processor_factory=lambda: processor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                    rtc_configuration=rtc_configuration,
                )
                
                # Display status
                if webrtc_ctx.state.playing:
                    st.success("✅ Webcam is active - object detection is running!")
                else:
                    st.warning("⚠️ Click 'START' to begin streaming.")
                
            except Exception as e:
                st.error(f"WebRTC error: {str(e)}")
                st.info("Try using the OpenCV method instead or check if streamlit-webrtc is installed.")
        else:
            # OpenCV method
            st.info("Select your webcam options below")
            
            webcam_options = {
                "Default Webcam": 0,
                "Secondary Webcam": 1
            }
            
            selected_webcam = st.selectbox("Select webcam", list(webcam_options.keys()))
            cam_source = webcam_options[selected_webcam]
            
            # Resolution selection
            resolution = st.selectbox(
                "Resolution",
                ["640x480", "800x600", "1280x720"],
                index=0
            )
            width, height = map(int, resolution.split('x'))
            
            # Store webcam state
            if 'webcam_running' not in st.session_state:
                st.session_state.webcam_running = False
            
            # Control buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if not st.session_state.webcam_running:
                    if st.button("▶️ Start Webcam", type="primary", use_container_width=True):
                        st.session_state.webcam_running = True
                        st.experimental_rerun()
            
            with col2:
                if st.session_state.webcam_running:
                    if st.button("⏹️ Stop Webcam", use_container_width=True):
                        st.session_state.webcam_running = False
                        st.experimental_rerun()
            
            # Create placeholder for webcam feed
            stframe = st.empty()
            
            if st.session_state.webcam_running:
                try:
                    # Open webcam
                    cap = cv2.VideoCapture(cam_source)
                    
                    # Set resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    
                    # Check if webcam opened successfully
                    if not cap.isOpened():
                        st.error(f"Failed to open webcam source: {cam_source}")
                        st.session_state.webcam_running = False
                        st.experimental_rerun()
                    
                    # Create temp file if saving video
                    temp_filename = None
                    out = None
                    
                    if save_video:
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        temp_filename = temp_file.name
                        
                        # Get webcam properties 
                        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = 20
                        
                        # Create VideoWriter object
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(temp_filename, fourcc, fps, (frame_width, frame_height))
                    
                    # Show webcam info
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    st.success(f"✅ Webcam connected! Resolution: {actual_width}x{actual_height}")
                    
                    # Initialize counters
                    frame_count = 0
                    error_count = 0
                    # Main webcam loop
                    st.info("⏳ Webcam streaming started. Processing frames...")
                    
                    # For FPS calculation
                    fps_calculation_interval = 5  # Calculate FPS every 5 frames
                    start_time = time.time()
                    fps = 0
                    
                    while st.session_state.webcam_running:
                        try:
                            # Read frame
                            ret, frame = cap.read()
                            
                            if not ret:
                                error_count += 1
                                if error_count > 5:  # Allow some errors before stopping
                                    st.error("❌ Failed to read from webcam. Stopping.")
                                    break
                                continue
                            
                            # Reset error counter on successful frame
                            error_count = 0
                            
                            # Process frame for object detection
                            boxes, scores, class_ids, processed_frame = detector.detect(frame)
                            
                            # Calculate FPS
                            frame_count += 1
                            if frame_count % fps_calculation_interval == 0:
                                end_time = time.time()
                                elapsed_time = end_time - start_time
                                fps = fps_calculation_interval / elapsed_time if elapsed_time > 0 else 0
                                start_time = time.time()
                            
                            # Display FPS on the frame
                            cv2.putText(
                                processed_frame, 
                                f"FPS: {fps:.1f}", 
                                (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (0, 255, 0), 
                                2
                            )
                            
                            # Write to video file if saving
                            if save_video and out is not None:
                                out.write(processed_frame)
                            
                            # Display frame
                            stframe.image(processed_frame, channels="BGR", use_column_width=True)
                            
                        except Exception as e:
                            st.error(f"Error processing frame: {e}")
                            break
                    
                    # Clean up
                    if cap is not None:
                        cap.release()
                    
                    if save_video and out is not None:
                        out.release()
                        
                        # Offer download option
                        with open(temp_filename, 'rb') as f:
                            video_bytes = f.read()
                        
                        st.download_button(
                            label="💾 Download Recorded Video",
                            data=video_bytes,
                            file_name="detection_video.mp4",
                            mime="video/mp4"
                        )
                        
                        # Clean up temp file
                        os.unlink(temp_filename)
                
                except Exception as e:
                    st.error(f"❌ Webcam error: {str(e)}")
                    st.session_state.webcam_running = False
    
    with tab3:
        st.header("About YOLO Vision Pro")
        
        st.markdown("""
        <div class="card">
            <h3>What is YOLO Vision Pro?</h3>
            <p>YOLO Vision Pro is an advanced object detection application powered by YOLOv8 (You Only Look Once), 
            a state-of-the-art computer vision model. This application allows you to detect and identify objects in 
            images and video streams in real-time.</p>
        </div>
        
        <div class="card">
            <h3>Features</h3>
            <ul>
                <li>🔍 <strong>Real-time object detection</strong> in images and webcam feeds</li>
                <li>📊 <strong>Multiple model options</strong> with different speed/accuracy tradeoffs</li>
                <li>⚙️ <strong>Adjustable detection parameters</strong> for fine-tuning results</li>
                <li>🎯 <strong>Accurate bounding boxes</strong> around detected objects</li>
                <li>📷 <strong>Support for webcam streaming</strong> with two different methods</li>
                <li>📥 <strong>Save and download</strong> processed images and videos</li>
            </ul>
        </div>
        
        <div class="card">
            <h3>How It Works</h3>
            <p>YOLO Vision Pro uses ONNX Runtime to run YOLOv8 models, providing fast and efficient inference across 
            different hardware platforms. The application:</p>
            <ol>
                <li>Takes input from an image upload or webcam</li>
                <li>Preprocesses the image to match model input requirements</li>
                <li>Runs the YOLOv8 model to detect objects</li>
                <li>Applies filters like confidence thresholds and NMS to refine detections</li>
                <li>Draws bounding boxes and labels for identified objects</li>
                <li>Displays or saves the processed results</li>
            </ol>
        </div>
        
        <div class="card">
            <h3>Available Models</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Size</th>
                    <th>Speed</th>
                    <th>Accuracy</th>
                    <th>Best For</th>
                </tr>
                <tr>
                    <td>YOLOv8 Nano</td>
                    <td>4.3 MB</td>
                    <td>⚡ Very Fast</td>
                    <td>Good</td>
                    <td>Mobile, edge devices, speed-critical applications</td>
                </tr>
                <tr>
                    <td>YOLOv8 Small</td>
                    <td>11.4 MB</td>
                    <td>🚀 Fast</td>
                    <td>Better</td>
                    <td>Balanced applications requiring both speed and accuracy</td>
                </tr>
                <tr>
                    <td>YOLOv8 Large</td>
                    <td>43.7 MB</td>
                    <td>Normal</td>
                    <td>🎯 Best</td>
                    <td>Accuracy-critical applications with adequate computing power</td>
                </tr>
            </table>
        </div>
        
        <div class="card">
            <h3>Requirements</h3>
            <p>To run this application, you need:</p>
            <ul>
                <li>Python 3.7 or higher</li>
                <li>ONNX Runtime</li>
                <li>OpenCV</li>
                <li>Streamlit</li>
                <li>Streamlit-WebRTC (for WebRTC webcam support)</li>
                <li>YOLOv8 ONNX model files</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Instructions section
        st.subheader("How to Use")
        
        with st.expander("📷 Image Upload Instructions"):
            st.markdown("""
            1. Navigate to the "Image Upload" tab
            2. Click "Choose an image..." to upload a photo
            3. Click "Detect Objects" to process the image
            4. View the results with bounding boxes and labels
            5. Download the processed image if desired
            """)
        
        with st.expander("🎥 Webcam Instructions"):
            st.markdown("""
            1. Navigate to the "Webcam" tab
            2. Choose between WebRTC or OpenCV methods
            3. Select your webcam and resolution settings
            4. Click "Start Webcam" to begin
            5. Watch as objects are detected in real-time
            6. Optional: Enable "Save Video" in the sidebar to record the session
            7. Click "Stop Webcam" when finished
            8. Download the recorded video if available
            """)


if __name__ == "__main__":
    main()
