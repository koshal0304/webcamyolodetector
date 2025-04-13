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
from streamlit_webrtc import RTCConfiguration, WebRtcMode
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
def load_custom_css():
    st.markdown("""
    <style>
    /* Mega Title Styling - UNCHANGED */
    .logo-text {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        font-family: 'Inter', sans-serif;
        background: linear-gradient(45deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        letter-spacing: -1.5px;
        margin: 0.5rem 0;
        line-height: 1.1;
        animation: titleGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        0% {
            text-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }
        100% {
            text-shadow: 0 8px 24px rgba(59, 130, 246, 0.5),
                         0 0 40px rgba(59, 130, 246, 0.2);
        }
    }
    
    /* Add floating animation - UNCHANGED */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .logo-container {
        animation: float 3s ease-in-out infinite;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        border-radius: 1rem;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
    }
    
    /* Subtitle styling - UNCHANGED */
    .logo-subtitle {
        font-size: 1.4rem !important;
        color: #94a3b8 !important;
        letter-spacing: 0.5px;
        margin-top: -0.5rem !important;
        font-weight: 400;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Enhanced Card styling for content sections */
    .stCard {
        border-radius: 1.25rem !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.4s ease !important;
        padding: 1.5rem !important;
    }
    
    .stCard:hover {
        transform: translateY(-7px);
        box-shadow: 0 25px 30px -12px rgba(0, 0, 0, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Improved Button styling */
    .stButton > button {
        border-radius: 0.75rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 20px -8px rgba(59, 130, 246, 0.5) !important;
    }
    
    .stButton > button:active {
        transform: translateY(1px) !important;
    }
    
    /* Refined form elements */
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 0.75rem !important;
        border-color: rgba(148, 163, 184, 0.25) !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03) !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
    }
    
    .stTextInput > div > div > input {
        border-radius: 0.75rem !important;
        border-color: rgba(148, 163, 184, 0.25) !important;
        transition: all 0.3s ease !important;
        padding: 0.75rem 1rem !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
    }
    
    /* Section headers - UNCHANGED */
    h2 {
        font-weight: 700 !important;
        color: #1e293b !important;
        letter-spacing: -0.5px !important;
        margin-top: 2rem !important;
        position: relative;
        display: inline-block;
    }
    
    h2:after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, transparent);
        border-radius: 3px;
    }
    
    /* Improved dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        h2 {
            color: #e2e8f0 !important;
        }
        
        .stCard {
            background-color: rgba(30, 41, 59, 0.7) !important;
        }
        
        .stSelectbox div[data-baseweb="select"] > div,
        .stTextInput > div > div > input {
            background-color: rgba(30, 41, 59, 0.6) !important;
            color: #e2e8f0 !important;
        }
    }
    
    /* Enhanced checkbox styling */
    .stCheckbox label {
        font-size: 1.05rem !important;
    }
    
    .stCheckbox label span {
        color: #475569 !important;
    }
    
    /* Enhanced custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(241, 245, 249, 0.15);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #3b82f6, #8b5cf6);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #2563eb, #7c3aed);
    }
    
    /* Enhanced background pattern */
    body {
        background-image: radial-gradient(circle at 1px 1px, rgba(99, 102, 241, 0.05) 1px, transparent 0) !important;
        background-size: 20px 20px !important;
    }
    
    /* Improve file uploader styling */
    [data-testid="stFileUploader"] {
        border-radius: 1rem !important;
        border: 2px dashed rgba(148, 163, 184, 0.3) !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(59, 130, 246, 0.5) !important;
        background-color: rgba(59, 130, 246, 0.03) !important;
    }
    
    /* Improve tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.1) !important;
        color: #3b82f6 !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)
def display_header():
    st.markdown("""
    <div class="logo-container">
        <div class="logo-text">YOLO Vision Pro</div>
        <div class="logo-subtitle">Advanced AI-Powered Object Detection</div>
    </div>
    """, unsafe_allow_html=True)

    
    st.markdown("""
    <p style="color: #64748b; margin-bottom: 2rem;">
        Advanced object detection powered by YOLOv8
    </p>
    """, unsafe_allow_html=True)
def style_sidebar():
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #1e3a8a; margin-bottom: 1rem;">Detection Settings</h2>
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
def create_results_card(image, boxes, scores, class_ids):
    # Display the image
    st.image(image, channels="BGR", use_container_width=True)
    
    # Create a summary card
    st.markdown("""
    <div class="detection-box">
        <h3 style="margin-top: 0;">Detection Summary</h3>
        <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
    """, unsafe_allow_html=True)
    
    # Get counts by class
    class_counts = {}
    for class_id in class_ids:
        class_name = COCO_LABELS[class_id] if 0 <= class_id < len(COCO_LABELS) else f"Unknown ({class_id})"
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Display top 5 detected objects
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for class_name, count in sorted_classes:
        st.markdown(f"""
        <div style="background-color: #f3f4f6; border-radius: 8px; padding: 0.5rem 1rem; display: inline-block;">
            <span style="font-weight: 600;">{class_name}</span>: {count}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# Streamlit UI
def main():
    # Set page config FIRST before any other Streamlit commands
    st.set_page_config(
        page_title="YOLO Vision Pro | Object Detection",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;900&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)
    # Load custom CSS
    load_custom_css()
    
    # Display header with branding
    display_header()
    
    # Rest of your code remains the same...
    # Style the sidebar
    style_sidebar()
    
    # Sidebar for configuration options
    st.sidebar.title("Model Settings")
    
    
    # Model selection with icons
    model_options = {
        "YOLOv8 Nano": "⚡ Fastest, 4.3MB",
        "YOLOv8 Small": "🚀 Fast, 11.4MB", 
        "YOLOv8 Medium": "⚖️ Balanced, 25.9MB",
        "YOLOv8 Large": "🎯 Most Accurate, 43.7MB"
    }
    
    model_option = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys()),
        format_func=lambda x: f"{x} ({model_options[x]})",
        index=0
    )
    
    # Map model options to file paths
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
        st.sidebar.error(f"⚠️ Model file not found: {model_path}")
        st.sidebar.info("📁 Please make sure to place the ONNX model files in the same directory as this script.")
        return
    
    # Add a separator
    st.sidebar.markdown("---")
    st.sidebar.subheader("Detection Parameters")
    
    # Detection parameters with tooltips
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.3, 
        step=0.05,
        help="Minimum confidence level for an object to be detected. Higher values reduce false positives."
    )
    
    iou_threshold = st.sidebar.slider(
        "IOU Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.45, 
        step=0.05,
        help="Intersection over Union threshold for Non-Maximum Suppression. Controls overlap of bounding boxes."
    )
    
    # Add a separator
    st.sidebar.markdown("---")
    st.sidebar.subheader("Output Options")
    
    # Save video option in sidebar
    save_video = st.sidebar.checkbox(
        "Save Video", 
        value=False,
        help="Enable to save detection video for later download"
    )
    
    # Create tabs for different input methods with icons
    tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "🎥 Webcam", "ℹ️ About"])
    
    with tab1:
        st.markdown("<h2>Upload Image</h2>", unsafe_allow_html=True)
        st.markdown("Upload an image to detect objects using YOLOv8")
        
        # Create a container for the uploader with custom styling
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Convert the file to an opencv image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Use columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h3>Original Image</h3>", unsafe_allow_html=True)
                st.image(image, channels="BGR", use_container_width=True)
            
            # Process button with better styling
            detect_btn = st.button(
                "🔍 Detect Objects", 
                type="primary",
                use_container_width=True
            )
            
            if detect_btn:
                with st.spinner("Processing image..."):
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
                        
                        with col2:
                            st.markdown("<h3>Detection Results</h3>", unsafe_allow_html=True)
                            st.image(processed_image, channels="BGR", use_container_width=True)
                        
                        # Display detection details using our helper function
                        display_detection_results(boxes, scores, class_ids)
                        
                        # Download options
                        if len(boxes) > 0:
                            # Option to download the processed image
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
                        st.info("Please try again with a different image or check the model configuration.")
    
    with tab2:
        st.markdown("<h2>Webcam Object Detection</h2>", unsafe_allow_html=True)
        st.markdown("Use your webcam to detect objects in real-time")
        
        # More visually appealing radio buttons
        webcam_method = st.radio(
            "Choose webcam method:",
            ["OpenCV (works best locally)", "WebRTC (better for deployed apps)"],
            horizontal=True
        )
        
        # Customize the webcam section for better usability
        # Replace your existing WebRTC implementation in the tab2 section with this code:

        if webcam_method == "WebRTC (better for deployed apps)":
            try:
                st.info("📹 Using browser-based WebRTC for webcam access. Allow camera permissions when prompted.")
                
                # Initialize the detector
                detector = YOLOv8Detector(
                    model_path=model_path,
                    labels=COCO_LABELS,
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold
                )
                
                # Create video processor class
                processor = VideoProcessor(detector)
                
                # Define RTC configuration with multiple STUN servers
                rtc_configuration = RTCConfiguration(
                    {"iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302", 
                                "stun:stun1.l.google.com:19302", 
                                "stun:stun2.l.google.com:19302"]}
                    ]}
                )
                
                # Create WebRTC streamer with improved configuration
                try:
                    webrtc_ctx = streamlit_webrtc.webrtc_streamer(
                        key="object-detection",
                        mode=WebRtcMode.SENDRECV,  # Explicit mode setting
                        rtc_configuration=rtc_configuration,
                        media_stream_constraints={"video": True, "audio": False},
                        video_processor_factory=lambda: processor,
                        async_processing=True,  # Try both True and False if issues persist
                    )
                    
                    # Show status with better formatting
                    if webrtc_ctx.state.playing:
                        st.success("✅ Webcam is streaming! Object detection is being applied to the video feed.")
                        
                        # Add a stats container
                        stats_container = st.container()
                        with stats_container:
                            st.markdown("""
                            <div class="detection-box">
                                <h3 style="margin-top: 0;">Detection Stats</h3>
                                <p>Real-time object detection is active. Detection results are shown directly in the video feed.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Click 'START' above to begin streaming from your webcam.")
                        st.info("💡 If the camera doesn't start, make sure you've granted camera permissions to this website.")
                except Exception as e:
                    st.error(f"WebRTC error: {str(e)}")
                    st.info("Try using the OpenCV webcam option instead, or check your browser's WebRTC compatibility.")
                    
                # Note about recording
                if save_video:
                    st.info("ℹ️ Video saving is not supported in WebRTC mode in this implementation.")
                
            except ImportError:
                st.error("❌ streamlit-webrtc package is not installed.")
                st.info("💡 To install it, run: pip install streamlit-webrtc")
                    
                # Note about recording
                if save_video:
                    st.info("ℹ️ Video saving is not supported in WebRTC mode in this implementation.")
                
            except ImportError:
                st.error("❌ streamlit-webrtc package is not installed.")
                st.info("💡 To install it, run: pip install streamlit-webrtc")
        # Update your OpenCV webcam implementation to be more deployment-friendly
            else:  # OpenCV implementation
                # Enhanced OpenCV-based implementation
                st.markdown("""
                <div class="detection-box">
                    <h3 style="margin-top: 0;">Webcam Configuration</h3>
                """, unsafe_allow_html=True)
                
                # Simplify webcam source - less options but more reliable in deployment
                webcam_source = st.selectbox(
                    "Webcam Source",
                    ["Default Webcam (0)"]
                )
                
                # Fixed resolution that works well in most cases
                width, height = 640, 480
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Store the webcam state
                if 'webcam_running' not in st.session_state:
                    st.session_state.webcam_running = False
                
                # Create visually appealing control buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if not st.session_state.webcam_running:
                        if st.button("▶️ Start Webcam", type="primary", use_container_width=True):
                            st.session_state.webcam_running = True
                            st.rerun()
                
                with col2:
                    if st.session_state.webcam_running:
                        if st.button("⏹️ Stop Webcam", type="secondary", use_container_width=True):
                            st.session_state.webcam_running = False
                            st.rerun()
                
                # Create placeholders for webcam feed and error messages
                stframe = st.empty()
                status_text = st.empty()
                
                if st.session_state.webcam_running:
                    try:
                        # Set source to 0 (default camera)
                        cam_source = 0
                        
                        # Open webcam with reliable settings for deployment
                        cap = cv2.VideoCapture(cam_source)
                        
                        # Set resolution (but be ready for it to be ignored)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        
                        # Check if webcam opened successfully
                        if not cap.isOpened():
                            st.error(f"❌ Failed to open webcam")
                            st.session_state.webcam_running = False
                            st.info("Your browser or deployment environment may not support webcam access.")
                            st.rerun()
                        
                        # Initialize detector
                        detector = YOLOv8Detector(
                            model_path=model_path,
                            labels=COCO_LABELS,
                            confidence_threshold=confidence_threshold,
                            iou_threshold=iou_threshold
                        )
                        
                        # Frame processing loop with better error handling
                        frame_count = 0
                        error_count = 0
                        max_errors = 5
                        
                        while st.session_state.webcam_running and error_count < max_errors:
                            ret, frame = cap.read()
                            
                            if not ret:
                                error_count += 1
                                status_text.warning(f"⚠️ Failed to grab frame ({error_count}/{max_errors})")
                                time.sleep(0.5)
                                continue
                            
                            # Process frame - with error handling
                            try:
                                boxes, scores, class_ids, processed_frame = detector.detect(frame)
                                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                                stframe.image(rgb_frame, channels="RGB", use_container_width=True)
                            except Exception as e:
                                status_text.error(f"Processing error: {str(e)}")
                                # Show original frame at least
                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                stframe.image(rgb_frame, channels="RGB", use_container_width=True)
                            
                            # Brief pause to avoid UI thread starvation
                            time.sleep(0.05)
                            frame_count += 1
                        
                        # Release resources
                        cap.release()
                        
                        if error_count >= max_errors:
                            st.error("❌ Webcam disconnected due to errors")
                            st.session_state.webcam_running = False
                            
                    except Exception as e:
                        st.error(f"❌ Webcam access error: {str(e)}")
                        st.info("Your browser or deployment environment may not support webcam access.")
                        st.session_state.webcam_running = False
                    
                    # Show detailed error with better formatting
                    st.markdown(f"""
                    <div class="detection-box">
                        <h3 style="margin-top: 0;">Error Details</h3>
                        <p>{str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<h2>About YOLO Vision Pro</h2>", unsafe_allow_html=True)
        
        # Two-column layout for about page
        about_col1, about_col2 = st.columns([2, 1])
        
        with about_col1:
            st.markdown("""
            ### What is YOLO Vision Pro?
            
            YOLO Vision Pro is a user-friendly web application for real-time object detection using the state-of-the-art YOLOv8 models. It allows you to detect objects in images and video streams with just a few clicks.
            
            ### Key Features:
            
            - **Multiple Models**: Choose from different YOLOv8 variants balancing speed and accuracy
            - **Real-time Detection**: Process webcam feeds in real-time
            - **Flexible Configuration**: Adjust confidence thresholds for better results
            - **Export Options**: Save processed images and videos
            - **Intuitive Interface**: User-friendly design for both beginners and experts
            
            ### How to Use:
            
            1. Select a model from the sidebar
            2. Set detection parameters as needed
            3. Upload an image or use your webcam
            4. View detection results and statistics
            
            ### About YOLOv8:
            
            YOLOv8 (You Only Look Once) is a state-of-the-art object detection algorithm that offers significant improvements in speed and accuracy compared to previous versions. It uses a single neural network to predict bounding boxes and class probabilities directly from full images in one evaluation.
            """)
            
            # Add expandable technical details
            with st.expander("📚 Technical Details"):
                st.markdown("""
                ### Implementation Details:
                
                - **ONNX Runtime**: This app uses ONNX Runtime for efficient model inference
                - **OpenCV Integration**: Uses OpenCV for image processing and visualization
                - **Streamlit Framework**: Built with Streamlit for a responsive web interface
                - **WebRTC Support**: Optional browser-based webcam capturing for better deployment compatibility
                
                ### Performance Considerations:
                
                - For best performance, run this application locally
                - Model loading time depends on the model size and your hardware
                - Detection speed varies based on input resolution and model choice
                - GPU acceleration is used when available
                """)
        
        with about_col2:
            # Display YOLOv8 architecture diagram or logo
            st.markdown("""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; text-align: center;">
                <h3>YOLOv8 Architecture</h3>
                <p>YOLOv8 uses a backbone-neck-head architecture:</p>
                <ul style="text-align: left;">
                    <li>Backbone: CSPDarknet</li>
                    <li>Neck: C2f modules</li>
                    <li>Head: Decoupled heads</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Add model comparison
            st.markdown("""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; margin-top: 20px;">
                <h3>Model Comparison</h3>
                <table style="width: 100%; text-align: center;">
                    <tr>
                        <th>Model</th>
                        <th>Size</th>
                        <th>Speed</th>
                        <th>Accuracy</th>
                    </tr>
                    <tr>
                        <td>Nano</td>
                        <td>4.3MB</td>
                        <td>⭐⭐⭐⭐⭐</td>
                        <td>⭐⭐</td>
                    </tr>
                    <tr>
                        <td>Small</td>
                        <td>11.4MB</td>
                        <td>⭐⭐⭐⭐</td>
                        <td>⭐⭐⭐</td>
                    </tr>
                    <tr>
                        <td>Medium</td>
                        <td>25.9MB</td>
                        <td>⭐⭐⭐</td>
                        <td>⭐⭐⭐⭐</td>
                    </tr>
                    <tr>
                        <td>Large</td>
                        <td>43.7MB</td>
                        <td>⭐⭐</td>
                        <td>⭐⭐⭐⭐⭐</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        # Credits section at bottom of about page
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #888;">
            <p>YOLO Vision Pro | Built with Streamlit and YOLOv8</p>
            <p>© 2024 | Open Source Project</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
