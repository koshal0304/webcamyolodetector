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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
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

# Now using VideoProcessorBase for better WebRTC compatibility
class VideoProcessor(VideoProcessorBase):
    def __init__(self, detector):
        self.detector = detector
        self.frame_counter = 0
        self.last_log_time = time.time()
        self.frames_processed = 0
        
    def recv(self, frame):
        self.frame_counter += 1
        self.frames_processed += 1
        
        # Log processing frequency
        current_time = time.time()
        time_diff = current_time - self.last_log_time
        if time_diff > 5.0:  # Log every 5 seconds
            fps = self.frames_processed / time_diff
            print(f"WebRTC processing FPS: {fps:.2f}")
            self.frames_processed = 0
            self.last_log_time = current_time
            
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Use your existing YOLOv8Detector
            boxes, scores, class_ids, processed_img = self.detector.detect(img)
            
            # Return processed frame
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        except Exception as e:
            print(f"Error in video processor: {e}")
            # Return original frame if error occurs
            return av.VideoFrame.from_ndarray(img, format="bgr24")

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


def main():
    # Load custom CSS
    load_custom_css()
    
    # Display header
    display_header()
    
    # Style sidebar
    style_sidebar()
    
    # App title
    st.title("YOLOv8 Object Detection App")
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📷 Webcam", "🖼️ Upload Image", "🎥 Upload Video"])
    
    # Sidebar configuration
    st.sidebar.markdown("## Model Settings")
    
    # Model selection (default to YOLOv8n for speed)
    model_options = {
        "YOLOv8n (Fast)": "models/yolov8n.onnx",
        "YOLOv8s (Balanced)": "models/yolov8s.onnx",
        "YOLOv8m (Accurate)": "models/yolov8m.onnx"
    }
    selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))
    model_path = model_options[selected_model]
    
    # Detection settings
    st.sidebar.markdown("## Detection Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please make sure you have the model files in the 'models' directory.")
        st.info("You can download YOLOv8 ONNX models from the official Ultralytics repository.")
        return
    
    # Initialize detector
    try:
        detector = YOLOv8Detector(
            model_path=model_path,
            labels=COCO_LABELS,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        st.sidebar.success(f"Model loaded successfully: {selected_model}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Webcam tab
    with tab1:
        st.header("Webcam Object Detection")
        st.markdown("Use your webcam for real-time object detection.")
        
        # WebRTC configuration
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Create a placeholder for the webrtc component
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            video_processor_factory=lambda: VideoProcessor(detector),
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if webrtc_ctx.state.playing:
            st.info("Webcam is active! Point your camera at objects to detect them.")
    
    # Image upload tab
    with tab2:
        st.header("Image Object Detection")
        st.markdown("Upload an image for object detection.")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display original image
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("Detect Objects"):
                with st.spinner("Processing..."):
                    # Run detection
                    boxes, scores, class_ids, processed_image = detector.detect(image)
                    
                    # Display processed image
                    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_column_width=True)
                    
                    # Display detection results
                    display_detection_results(boxes, scores, class_ids)
    
    # Video upload tab
    with tab3:
        st.header("Video Object Detection")
        st.markdown("Upload a video for object detection.")
        
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
        
        if uploaded_video is not None:
            # Save uploaded video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            # Get video info
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            # Display video information
            st.write(f"Video FPS: {fps}")
            st.write(f"Duration: {duration:.2f} seconds")
            st.write(f"Total Frames: {frame_count}")
            
            # Processing options
            st.markdown("### Processing Options")
            
            processing_fps = st.slider("Processing FPS", 1, fps, min(fps, 15))
            frame_step = max(1, int(fps / processing_fps))
            
            # Start processing button
            if st.button("Process Video"):
                # Create progress bar
                progress_bar = st.progress(0)
                
                # Create a placeholder for the video frames
                frame_placeholder = st.empty()
                
                # Process video frames
                frame_idx = 0
                processed_frames = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process only every nth frame based on frame_step
                    if frame_idx % frame_step == 0:
                        # Update progress
                        progress = min(float(frame_idx) / frame_count, 1.0)
                        progress_bar.progress(progress)
                        
                        # Run detection
                        boxes, scores, class_ids, processed_frame = detector.detect(frame)
                        
                        # Convert to RGB for display
                        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display the processed frame
                        frame_placeholder.image(processed_frame_rgb, caption=f"Processing: Frame {frame_idx}", use_column_width=True)
                        
                        # Add to processed frames list (for future saving if needed)
                        processed_frames.append(processed_frame)
                    
                    frame_idx += 1
                
                # Release video
                cap.release()
                
                # Clean up temp file
                os.unlink(video_path)
                
                st.success("Video processing complete!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>Powered by YOLOv8 and ONNX Runtime</p>
        <p style="font-size: 0.8rem; color: #64748b;">
            This app uses state-of-the-art object detection to identify and locate objects in images and videos.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
