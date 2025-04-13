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
import streamlit_webrtc
# Import your existing YOLOv8Detector class
# This is the exact same class from your code, unchanged
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
        
        # Initialize ONNX runtime session
        # Try with CUDA first, fallback to CPU if CUDA not available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"Using providers: {self.session.get_providers()}")
        except Exception as e:
            print(f"Failed to use CUDA, falling back to CPU: {e}")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            print("Using CPU for inference")
        
        # Get model info
        self.get_input_details()
        self.get_output_details()
        print(f"Model loaded successfully. Input shape: {self.input_shape}")
        
        # Print model output shape to debug
        outputs = self.session.get_outputs()
        print(f"Model output shape: {outputs[0].shape}")
    
    def get_input_details(self):
        """Get model input details"""
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        
        self.input_shape = model_inputs[0].shape
        self.channels = 3  # Usually fixed at 3 for RGB
        
        print(f"Setting input dimensions to: {self.input_width}x{self.input_height}")
    
    def get_output_details(self):
        """Get model output details"""
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        
        # Print output names for debugging
        print(f"Output names: {self.output_names}")
    
    def prepare_input(self, image):
        """
        Prepare image for inference
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
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
        
        Args:
            outputs: Model outputs
            
        Returns:
            boxes, scores, class_ids
        """
        # YOLOv8 has a different output format than YOLOv5/v7
        # The output shape is typically [1, 84, num_boxes] where:
        # - The first 4 values (0-3) are the box coordinates (x, y, w, h)
        # - The rest (4-83) are the confidence scores for each class
        
        # Print output shape for debugging
        output = outputs[0]
        print(f"Raw output shape: {output.shape}")
        
        # Handle different possible output formats
        if len(output.shape) == 3:  # [batch, boxes, elements]
            predictions = np.squeeze(output)  # Remove batch dimension
            
            # YOLOv8 format
            if predictions.shape[0] == 84 or predictions.shape[0] > 84:
                # Transpose to have predictions in expected format [num_boxes, elements]
                predictions = predictions.transpose(1, 0)
            
            print(f"Processed output shape: {predictions.shape}")
            
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
                    if isinstance(indices[0], np.ndarray):  # OpenCV 4.5.4+ returns 1D array
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
        # This is similar to the original code
        else:
            print("Using fallback detection method")
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
                        if isinstance(indices[0], np.ndarray):
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
        """
        Extract boxes from predictions
        
        Args:
            predictions: Predictions from the model
            
        Returns:
            Boxes in pixels
        """
        # Extract boxes from predictions
        boxes = predictions[:, :4]
        
        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)
        
        # Convert boxes to xyxy format (xmin, ymin, xmax, ymax)
        boxes = self.xywh2xyxy(boxes)
        
        return boxes
    
    def rescale_boxes(self, boxes):
        """
        Rescale boxes to original image dimensions
        
        Args:
            boxes: Boxes from the model (xywh or xyxy)
            
        Returns:
            Rescaled boxes
        """
        # Rescale boxes to original image dimensions
        scale_x = self.img_width / self.input_width
        scale_y = self.img_height / self.input_height
        
        # Scale each coordinate appropriately
        scaled_boxes = boxes.copy()
        scaled_boxes[:, 0] *= scale_x  # x1 or x
        scaled_boxes[:, 1] *= scale_y  # y1 or y
        scaled_boxes[:, 2] *= scale_x  # x2 or w
        scaled_boxes[:, 3] *= scale_y  # y2 or h
        
        return scaled_boxes
    
    def xywh2xyxy(self, x):
        """
        Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
        
        Args:
            x: Boxes in xywh format
            
        Returns:
            Boxes in xyxy format
        """
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    def draw_detections(self, image, boxes, scores, class_ids):
        """
        Draw detections on the image
        
        Args:
            image: The input image
            boxes: Bounding boxes
            scores: Confidence scores
            class_ids: Class indices
            
        Returns:
            Image with detections
        """
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
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Get size of the text
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Make sure text background stays within image bounds
            if y1 - text_height - 5 < 0:
                # If label would be above the image, put it inside the bounding box
                text_y = y1 + text_height + 5
                cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 + text_height + 5), color, -1)
                cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            else:
                # Draw filled rectangle for text background
                cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
                # Add text
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return image
    
    def detect(self, image):
        """
        Detect objects in the input image
        
        Args:
            image: The input image
            
        Returns:
            boxes, scores, class_ids: Detection results
            processed_image: Image with detections
        """
        # Create a copy of the image for drawing
        processed_image = image.copy()
        
        # Prepare input
        input_tensor = self.prepare_input(image)
        
        # Perform inference
        start = time.time()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        end = time.time()
        
        # Debug output shape
        print(f"Output shape: {outputs[0].shape}")
        
        # Process output
        boxes, scores, class_ids = self.process_output(outputs)
        
        print(f"Detections: {len(boxes)} objects found")
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            class_name = self.labels[class_id] if 0 <= class_id < len(self.labels) else f"Unknown ({class_id})"
            print(f"  {i+1}. {class_name}: {score:.2f} at {box}")
        
        # Draw detections
        processed_image = self.draw_detections(processed_image, boxes, scores, class_ids)
        
        # Calculate inference time
        inference_time = end - start
        fps = 1 / inference_time if inference_time > 0 else 0
        
        # Display FPS
        cv2.putText(processed_image, f'FPS: {fps:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return boxes, scores, class_ids, processed_image

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
class VideoProcessor:
    def __init__(self, detector):
        self.detector = detector
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Use your existing YOLOv8Detector
        boxes, scores, class_ids, processed_img = self.detector.detect(img)
        
        # Return processed frame
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
# Streamlit UI
def main():
    st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
    
    # Add a custom logo or header
    st.title("YOLOv8 Object Detection")
    st.subheader("Upload an image or use your webcam to detect objects")
    
    # Sidebar for configuration options
    st.sidebar.title("Settings")
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Select Model",
        ["YOLOv8 Nano", "YOLOv8 Small", "YOLOv8 Medium", "YOLOv8 Large"],
        index=0
    )
    
    # Map model options to file paths (assuming the models are in the same directory)
    model_paths = {
        "YOLOv8 Nano": "yolov8n.onnx",
        "YOLOv8 Small": "yolov8s.onnx",
        "YOLOv8 Medium": "yolov8m.onnx",
        "YOLOv8 Large": "yolov8l.onnx"
    }
    
    # Get selected model path
    model_path = model_paths[model_option]
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model file not found: {model_path}")
        st.sidebar.info("Please make sure to place the ONNX model files in the same directory as this script.")
        return
    
    # Detection parameters
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
    iou_threshold = st.sidebar.slider("IOU Threshold", 0.1, 1.0, 0.45, 0.05)
    
    # Save video option in sidebar (only defined once)
    save_video = st.sidebar.checkbox("Save Video", value=False)
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Image Upload", "Webcam", "About"])
    
    with tab1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Convert the file to an opencv image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display original image
            st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)
            
            # Process button
            if st.button("Detect Objects"):
                with st.spinner("Processing..."):
                    try:
                        # Initialize the detector
                        detector = YOLOv8Detector(
                            model_path=model_path,
                            labels=COCO_LABELS,
                            confidence_threshold=confidence_threshold,
                            iou_threshold=iou_threshold
                        )
                        
                        # Perform detection
                        boxes, scores, class_ids, processed_image = detector.detect(image)
                        
                        # Display results
                        st.image(processed_image, channels="BGR", caption="Detection Results", use_container_width=True)
                        
                        # Display detection details
                        if len(boxes) > 0:
                            st.success(f"Found {len(boxes)} objects!")
                            
                            # Create a dataframe to display results
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
                            
                            # Option to download the processed image
                            processed_img = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(processed_img)
                            buf = io.BytesIO()
                            pil_img.save(buf, format="PNG")
                            
                            st.download_button(
                                label="Download Processed Image",
                                data=buf.getvalue(),
                                file_name="detection_result.png",
                                mime="image/png"
                            )
                        else:
                            st.info("No objects detected.")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Modify the webcam section in tab2 to handle webcam access errors more gracefully

    with tab2:
        st.header("Webcam Object Detection")
        webcam_method = st.radio(
        "Choose webcam method:",
        ["OpenCV (works best locally)", "WebRTC (better for deployed apps)"]
    )
    
    if webcam_method == "WebRTC (better for deployed apps)":
        try:
            # Check if the streamlit-webrtc package is available
            import streamlit_webrtc
            
            st.info("Using browser-based WebRTC for webcam access. Allow camera permissions when prompted.")
            
            # Initialize the detector
            detector = YOLOv8Detector(
                model_path=model_path,
                labels=COCO_LABELS,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )
            
            # Create video processor class
            processor = VideoProcessor(detector)
            
            # Create WebRTC streamer
            webrtc_ctx = streamlit_webrtc.webrtc_streamer(
                key="object-detection",
                video_processor_factory=lambda: processor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            # Show status
            if webrtc_ctx.state.playing:
                st.success("Webcam is streaming! Object detection is being applied to the video feed.")
            else:
                st.warning("Click 'START' above to begin streaming from your webcam.")
                
            # Note about recording (currently not supported in this simple implementation)
            if save_video:
                st.warning("Video saving is not supported in WebRTC mode in this implementation.")
                
        except ImportError:
            st.error("streamlit-webrtc package is not installed.")
            st.info("To install it, run: pip install streamlit-webrtc")
            
    else:
        # Your original OpenCV-based implementation here
        # (The code from the first artifact)
        # Add webcam selection options
        st.sidebar.subheader("Webcam Options")
        webcam_source = st.sidebar.selectbox(
            "Webcam Source",
            ["Default Webcam (0)", "Secondary Webcam (1)", "Custom Device Path", "Browser-based (Streamlit Component)"],
            index=0
        )
        
        if webcam_source == "Custom Device Path":
            custom_path = st.sidebar.text_input("Device Path (e.g., /dev/video0)", "/dev/video0")
        
        # Add resolution settings
        webcam_resolution = st.sidebar.selectbox(
            "Webcam Resolution",
            ["640x480", "800x600", "1280x720", "Custom"],
            index=0
        )
        
        if webcam_resolution == "Custom":
            col1, col2 = st.sidebar.columns(2)
            with col1:
                width = st.number_input("Width", min_value=320, max_value=1920, value=640, step=10)
            with col2:
                height = st.number_input("Height", min_value=240, max_value=1080, value=480, step=10)
        else:
            width, height = map(int, webcam_resolution.split('x'))
        
        # Store the webcam state
        if 'webcam_running' not in st.session_state:
            st.session_state.webcam_running = False
        
        # Create buttons to control webcam
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.webcam_running:
                if st.button("Start Webcam"):
                    st.session_state.webcam_running = True
                    st.session_state.webcam_error = None
                    st.rerun()
        
        with col2:
            if st.session_state.webcam_running:
                if st.button("Stop Webcam"):
                    st.session_state.webcam_running = False
                    st.rerun()
        
        # Display information about browser permissions
        st.info("Make sure to allow camera permissions in your browser when prompted.")
        
        # Create placeholders for webcam feed and error messages
        stframe = st.empty()
        status_text = st.empty()
        
        if st.session_state.webcam_running:
            try:
                # Determine the webcam source
                if webcam_source == "Default Webcam (0)":
                    cam_source = 0
                elif webcam_source == "Secondary Webcam (1)":
                    cam_source = 1
                elif webcam_source == "Custom Device Path":
                    cam_source = custom_path
                elif webcam_source == "Browser-based (Streamlit Component)":
                    # Use streamlit-webrtc if available (need to be installed)
                    st.error("Please install streamlit-webrtc package to use browser-based webcam")
                    st.code("pip install streamlit-webrtc")
                    st.session_state.webcam_running = False
                    st.rerun()
                
                # Open webcam with specified settings
                cap = cv2.VideoCapture(cam_source)
                
                # Set resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Check if webcam opened successfully
                if not cap.isOpened():
                    st.error(f"Failed to open webcam source: {cam_source}")
                    st.session_state.webcam_running = False
                    
                    # Provide more helpful diagnostics
                    st.markdown("""
                    ### Troubleshooting:
                    1. Try a different webcam source
                    2. Check if your camera is being used by another application
                    3. Try running the app locally instead of deployed
                    4. On Linux, check device permissions with `ls -l /dev/video*`
                    """)
                    
                    # Try listing available cameras (Linux)
                    try:
                        import subprocess
                        video_devices = subprocess.check_output("ls -l /dev/video*", shell=True).decode()
                        st.code(video_devices)
                    except:
                        pass
                    
                    st.rerun()
                
                # Initialize detector
                detector = YOLOv8Detector(
                    model_path=model_path,
                    labels=COCO_LABELS,
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold
                )
                
                # Create a temporary file to save video if needed
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
                st.success(f"Webcam connected! Resolution: {actual_width}x{actual_height}")
                
                # Initialize frame counter and error counter
                frame_count = 0
                error_count = 0
                max_errors = 5
                
                # Main webcam loop
                while st.session_state.webcam_running and error_count < max_errors:
                    ret, frame = cap.read()
                    
                    if not ret:
                        error_count += 1
                        st.warning(f"Failed to grab frame ({error_count}/{max_errors})")
                        time.sleep(0.5)
                        continue
                    
                    # Reset error counter on successful frame
                    error_count = 0
                    frame_count += 1
                    
                    try:
                        # Detect objects
                        boxes, scores, class_ids, processed_frame = detector.detect(frame)
                        
                        # Save frame if recording
                        if save_video and out is not None:
                            out.write(processed_frame)
                        
                        # Convert to RGB for display
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display frame
                        stframe.image(rgb_frame, channels="RGB", use_container_width=True)
                        
                        # Show detection count
                        status_text.text(f"Frame: {frame_count} | Detected {len(boxes)} objects")
                        
                    except Exception as e:
                        st.error(f"Error during detection: {e}")
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        stframe.image(rgb_frame, channels="RGB", use_container_width=True)
                    
                    # Add slight delay to not overload the CPU
                    time.sleep(0.01)
                
                # Release resources
                cap.release()
                if save_video and out is not None:
                    out.release()
                    
                    # Offer download link for recorded video
                    with open(temp_filename, 'rb') as f:
                        st.download_button(
                            label="Download Recorded Video", 
                            data=f,
                            file_name="detection_video.mp4",
                            mime="video/mp4"
                        )
                
                # If stopped due to errors
                if error_count >= max_errors:
                    st.error("Stopped webcam due to repeated frame capture errors.")
                    st.session_state.webcam_running = False
                
            except Exception as e:
                st.error(f"Webcam error: {str(e)}")
                st.session_state.webcam_running = False
                
                # Provide troubleshooting help
                st.markdown("""
                ### Webcam Troubleshooting:
                
                1. **Browser permissions**: Make sure your browser has permission to access the camera
                2. **Hardware access**: In cloud deployments, webcam access may be limited
                3. **Alternative approach**: Consider installing the alternative package for browser-based capture:
                """)
                st.code("pip install streamlit-webrtc")
                
                # Add option for trying an alternative method
                if st.button("Try Alternative Method"):
                    st.session_state.try_alt_method = True
                    st.rerun()
        
        # Show a placeholder message when webcam is not running
        if not st.session_state.webcam_running:
            stframe.info("Click 'Start Webcam' to begin detection")
            
        # Add browser-based fallback option using streamlit-webrtc if installed
        if 'try_alt_method' in st.session_state and st.session_state.try_alt_method:
            st.subheader("Alternative Webcam Method")
            try:
                import streamlit_webrtc
                st.success("streamlit-webrtc is installed!")
                st.info("This component allows webcam access via browser APIs, which may work better in deployed environments.")
                # Implementation would go here
            except ImportError:
                st.error("streamlit-webrtc is not installed. Install it with: pip install streamlit-webrtc")
    with tab3:
        st.header("About")
        st.markdown("""
        ## YOLOv8 Object Detection App
        
        This application uses the YOLOv8 object detection model converted to ONNX format for efficient inference.
        
        ### Features:
        - Support for multiple YOLOv8 model variants
        - Image upload and webcam support
        - Adjustable confidence and IoU thresholds
        - Detection visualization with bounding boxes
        - Detection statistics
        - Option to save and download results
        
        ### How to use:
        1. Select a model from the sidebar
        2. Adjust detection parameters if needed
        3. Upload an image or use your webcam
        4. View detection results
        
        ### Requirements:
        - OpenCV
        - ONNX Runtime
        - Streamlit
        - NumPy
        - Pillow
        
        ### Models:
        Place the YOLOv8 ONNX models in the same directory as this script:
        - yolov8n.onnx (Nano version)
        - yolov8s.onnx (Small version)
        - yolov8m.onnx (Medium version)
        - yolov8l.onnx (Large version)
        """)

if __name__ == "__main__":
    main()
