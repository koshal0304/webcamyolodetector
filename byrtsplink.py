import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import argparse
from threading import Thread
from collections import deque

# Keep all the existing classes (TemporalSmoother, RTSPCapture, etc.)
# I'll only show the modified parts and new additions 
class TemporalSmoother:
    """Advanced temporal smoother for bounding boxes to reduce flickering"""
    def __init__(self, history_size=7, position_alpha=0.3, size_alpha=0.2, score_alpha=0.5, 
                 iou_threshold=0.25, min_hits=2, max_age=12):
        self.history_size = history_size
        self.position_alpha = position_alpha
        self.size_alpha = size_alpha
        self.score_alpha = score_alpha
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 0
        self.last_frame_tracks = []
          # Store the tracks from the last frame for consistency
    
    def update(self, boxes, scores, class_ids):
        """Update tracked objects with new detections"""
        # Convert inputs to lists if they're numpy arrays
        if isinstance(boxes, np.ndarray):
            boxes = boxes.tolist()
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        if isinstance(class_ids, np.ndarray):
            class_ids = class_ids.tolist()   
        # If no new detections provided
        if not boxes:
            # If no tracks exist yet, return empty results
            if not self.tracks:
                self.last_frame_tracks = ([], [], [])
                return [], [], []
            
            # Increment age for all tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                
                # For tracks without new detections, apply slight dampening to size
                # This helps prevent boxes from "growing" when object is occluded
                if self.tracks[track_id]['age'] > 2:
                    box = self.tracks[track_id]['smooth_box']
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    center_x = (box[0] + box[2]) / 2 
                    center_y = (box[1] + box[3]) / 2
            
                    # Slightly reduce box size for occluded objects (0.99 factor)
                    dampened_width = width * 0.99
                    dampened_height = height * 0.99 
                    
                    # Recalculate box with dampened size
                    box[0] = center_x - dampened_width / 2
                    box[1] = center_y - dampened_height / 2
                    box[2] = center_x + dampened_width / 2
                    box[3] = center_y + dampened_height / 2
                    
                    self.tracks[track_id]['smooth_box'] = box
                
                # Remove tracks that are too old
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
            
            # Get predictions for remaining tracks
            result = self._get_predictions()
            self.last_frame_tracks = result
            return result
        
        # With new detections, match them to existing tracks
        matched_tracks = {}
        matched_detections = set()
        
        # First match based on IoU and class
        for track_id, track in self.tracks.items():
            best_iou = self.iou_threshold
            best_detection_idx = -1
            
            track_box = track['smooth_box'] 
            
            for i, box in enumerate(boxes):
                if i not in matched_detections:
                    iou = self._calculate_iou(track_box, box)
                    if iou > best_iou and track['class_id'] == class_ids[i]:
                        best_iou = iou
                        best_detection_idx = i
            
            if best_detection_idx >= 0:
                # Update existing track
                self._update_track(track, 
                                  boxes[best_detection_idx], 
                                  scores[best_detection_idx],
                                  class_ids[best_detection_idx])
                
                matched_tracks[track_id] = track
                matched_detections.add(best_detection_idx)
                track['age'] = 0  # Reset age
                track['hits'] += 1 
            else:
                # Track not matched but keep it alive for a few frames
                track['age'] += 1
                
                # Keep track if it's not too old
                if track['age'] <= self.max_age:
                    matched_tracks[track_id] = track
        
        # Create new tracks for unmatched detections
        for i in range(len(boxes)):
            if i not in matched_detections:
                # Create new track
                self.tracks[self.next_id] = {
                    'box_history': [boxes[i]] * self.history_size,  # Initialize history with current box
                    'smooth_box': boxes[i],  # Start with current box
                    'score': scores[i],
                    'class_id': class_ids[i],
                    'age': 0,
                    'hits': 1,
                    'visible': False  # Start invisible until min_hits is reached
                }
                matched_tracks[self.next_id] = self.tracks[self.next_id]
                self.next_id += 1 
    
        # Update tracked objects
        self.tracks = {k: v for k, v in matched_tracks.items()}
        
        # Get predictions
        result = self._get_predictions()
        self.last_frame_tracks = result
        return result
    
    def _update_track(self, track, box, score, class_id):
        """Update a track with a new detection"""
        # Update box history
        track['box_history'].pop(0)
        track['box_history'].append(box)
        
        # Smooth bounding box position and size
        x1, y1, x2, y2 = box
        sx1, sy1, sx2, sy2 = track['smooth_box']
        
        # Calculate centers
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        s_center_x = (sx1 + sx2) / 2
        s_center_y = (sy1 + sy2) / 2
        
        # Calculate dimensions
        width = x2 - x1 
        height = y2 - y1 
        s_width = sx2 - sx1 
        s_height = sy2 - sy1 
        
        # Apply different smoothing factors to position vs. size
        # Smooth center position
        new_center_x = (1 - self.position_alpha) * s_center_x + self.position_alpha * center_x
        new_center_y = (1 - self.position_alpha) * s_center_y + self.position_alpha * center_y
        
        # Apply size smoothing
        new_width = (1 - self.size_alpha) * s_width + self.size_alpha * width
        new_height = (1 - self.size_alpha) * s_height + self.size_alpha * height
        
        # Recalculate box from center and dimensions
        new_x1 = new_center_x - new_width / 2
        new_y1 = new_center_y - new_height / 2
        new_x2 = new_center_x + new_width / 2
        new_y2 = new_center_y + new_height / 2
        
        # Update smoothed box
        track['smooth_box'] = [new_x1, new_y1, new_x2, new_y2]
        
        # Smooth score - more aggressive smoothing for scores to prevent flickering
        track['score'] = (1 - self.score_alpha) * track['score'] + self.score_alpha * score
        
        # Class ID usually doesn't change but we update it
        track['class_id'] = class_id
        
        # Mark as visible once it has enough hits
        if track['hits'] >= self.min_hits:
            track['visible'] = True 
    
    def _get_predictions(self):
        """Get current predictions from tracks"""
        boxes = []
        scores = []
        class_ids = []
        
        for track_id, track in self.tracks.items():
            # Only include tracks with enough hits or already visible
            if track['visible'] or track['hits'] >= self.min_hits:
                track['visible'] = True  # Once visible, stay visible
                boxes.append(track['smooth_box'])
                scores.append(track['score'])
                class_ids.append(track['class_id'])
        
        return boxes, scores, class_ids
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        # Convert to float to avoid integer division issues
        x1_1, y1_1, x2_1, y2_1 = map(float, box1)
        x1_2, y1_2, x2_2, y2_2 = map(float, box2)
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
class RTSPCapture:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.capture = cv2.VideoCapture(rtsp_url)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.thread = Thread(target=self.update, daemon=True)
        self.status = False
        self.frame = None
        self.running = True
        self.thread.start()
    
    def update(self): 
        while self.running:
            if self.capture.isOpened():
                (self.status, frame) = self.capture.read()
                if self.status:
                    self.frame = frame
            else:
                print("Reconnecting to RTSP stream...")
                self.capture = cv2.VideoCapture(self.rtsp_url)
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(0.01)  # Short sleep to prevent CPU overload
    
    def read(self):
        return self.status, self.frame if self.status else None
    
    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.capture.release()

class PolygonROI:
    """Class to handle polygon-based region of interest"""
    def __init__(self):
        self.points = []
        self.is_drawing = False
        self.is_complete = False
        
    def add_point(self, x, y):
        """Add a point to the polygon"""
        if not self.is_complete:
            self.points.append((x, y))
            
    def complete_polygon(self):
        """Complete the polygon by closing it"""
        if len(self.points) >= 3:
            self.is_complete = True
            return True
        return False
    
    def is_point_inside(self, x, y):
        """Check if a point is inside the polygon using point-in-polygon algorithm"""
        if not self.is_complete or len(self.points) < 3:
            return True  # If polygon is not complete, consider all points inside
            
        # Convert points to numpy array for cv2.pointPolygonTest
        polygon = np.array(self.points, dtype=np.int32)
        result = cv2.pointPolygonTest(polygon, (x, y), False)
        return result >= 0
    
    def is_box_inside(self, box):
        """Check if a bounding box is inside the polygon
        
        Args:
            box: [x1, y1, x2, y2] bounding box coordinates
            
        Returns:
            True if the center of the box is inside the polygon
        """
        if not self.is_complete:
            return True  # If polygon is not complete, consider all boxes inside
            
        # Calculate center point of the box
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2

        
        # Check if center point is inside polygon
        return self.is_point_inside(center_x, center_y)
    
    def draw(self, image):
        """Draw the polygon on the image"""
        if len(self.points) > 0:
            # Draw polygon points
            for point in self.points:
                cv2.circle(image, point, 5, (0, 255, 0), -1)
            
            # Draw polygon lines
            if len(self.points) > 1:
                for i in range(len(self.points) - 1):
                    cv2.line(image, self.points[i], self.points[i+1], (0, 255, 0), 2)
                
                # Draw closing line if polygon is complete
                if self.is_complete:
                    cv2.line(image, self.points[-1], self.points[0], (0, 255, 0), 2)
            
            # Draw semi-transparent fill if polygon is complete
            if self.is_complete and len(self.points) >= 3:
                overlay = image.copy()
                polygon = np.array(self.points, dtype=np.int32)
                cv2.fillPoly(overlay, [polygon], (0, 255, 0, 64))  # Green with low alpha
                cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)  # Blend with original image
                
        return image
    
    def save(self, filename):
        """Save polygon points to a file"""
        with open(filename, 'w') as f:
            for point in self.points:
                f.write(f"{point[0]},{point[1]}\n")
        print(f"Polygon saved to {filename}")
    
    def load(self, filename):
        """Load polygon points from a file"""
        if not os.path.exists(filename):
            print(f"Polygon file {filename} not found.")
            return False
            
        try:
            self.points = []
            with open(filename, 'r') as f:
                for line in f:
                    x, y = map(int, line.strip().split(','))
                    self.points.append((x, y))
            
            if len(self.points) >= 3:
                self.is_complete = True
                print(f"Polygon loaded from {filename}")
                return True
            else:
                print(f"Invalid polygon in {filename} (needs at least 3 points)")
                return False
        except Exception as e:
            print(f"Error loading polygon: {e}")
            return False


# Modify the YOLOv8Detector.draw_detections method to filter boxes by ROI
class YOLOv8Detector:
    # ... (keep all existing methods)
    def __init__(self, model_path, labels, confidence_threshold=0.45, iou_threshold=0.45, input_width=320, input_height=320):
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
        
        # Initialize ONNX runtime session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 4  # Adjust based on your CPU cores
        
        # Try with CUDA first, fallback to CPU if CUDA not available
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
            print(f"Using providers: {self.session.get_providers()}")
        except Exception as e:
            print(f"Failed to use CUDA, falling back to CPU: {e}")
            self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
            print("Using CPU for inference")
        
        # Get model info
        self.get_input_details()
        self.get_output_details()
        print(f"Model loaded successfully. Input shape: {self.input_shape}")
        
        # Print model output shape to debug
        outputs = self.session.get_outputs()
        print(f"Model output shape: {outputs[0].shape}")

        # Initialize object tracker
        self.tracker = TemporalSmoother(
            history_size=7,  # Increase history size for smoother tracking
            position_alpha=0.2,  # Decrease for smoother motion (0.3 is more stable)
            size_alpha=0.2,    # Lower value for smoother size changes
            score_alpha=0.5,   # Balance between responsiveness and stability
            iou_threshold=0.35, # Slightly lower to better match objects across frames
            min_hits=2,        # Require at least 2 hits before showing a track
            max_age=12)    
    
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
        input_img = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        
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
    
    def draw_detections(self, image, boxes, scores, class_ids, roi=None):
        """
        Draw detections on the image, filtered by region of interest
        
        Args:
            image: The input image
            boxes: Bounding boxes
            scores: Confidence scores
            class_ids: Class indices
            roi: PolygonROI instance for filtering
            
        Returns:
            Image with detections
        """
        # Create a copy of the image to avoid modifying the original
        output_image = image.copy()
        
        # Set fixed thickness for consistent appearance
        box_thickness = 2
        text_thickness = 1
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            # Filter by ROI if provided
            if roi is not None and not roi.is_box_inside(box):
                continue  # Skip boxes outside the ROI
            
            # Ensure class_id is within the valid range
            if class_id < 0 or class_id >= len(self.labels):
                label = f'Unknown ({class_id}): {score:.2f}'
            else:
                label = f'{self.labels[class_id]}: {score:.2f}'
            
            # Make sure all box coordinates are integers and round consistently
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Make sure coordinates are within image bounds
            x1 = max(0, min(x1, output_image.shape[1] - 1))
            y1 = max(0, min(y1, output_image.shape[0] - 1))
            x2 = max(0, min(x2, output_image.shape[1] - 1))
            y2 = max(0, min(y2, output_image.shape[0] - 1))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Use a fixed color map instead of HSV conversion to ensure consistency
            # This creates a deterministic color based on class_id
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0),
                (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128)
            ]
            color = colors[class_id % len(colors)]
            
            # Draw rectangle with fixed thickness
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, box_thickness)
            
            # Get size of the text with fixed font parameters
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            
            # Add padding to text background for better appearance
            text_padding = 5
            
            # Make sure text background stays within image bounds
            if y1 - text_height - (2 * text_padding) < 0:
                # If label would be above the image, put it inside the bounding box
                text_y = y1 + text_height + text_padding
                # Draw filled rectangle for text background
                cv2.rectangle(
                    output_image, 
                    (x1, y1), 
                    (x1 + text_width + (2 * text_padding), y1 + text_height + (2 * text_padding)), 
                    color, 
                    -1
                )
                # Add text
                cv2.putText(
                    output_image, 
                    label, 
                    (x1 + text_padding, text_y), 
                    font, 
                    font_scale, 
                    (0, 0, 0), 
                    text_thickness
                )
            else:
                # Draw filled rectangle for text background
                cv2.rectangle(
                    output_image, 
                    (x1, y1 - text_height - (2 * text_padding)), 
                    (x1 + text_width + (2 * text_padding), y1), 
                    color, 
                    -1
                )
                # Add text
                cv2.putText(
                    output_image, 
                    label, 
                    (x1 + text_padding, y1 - text_padding), 
                    font, 
                    font_scale, 
                    (0, 0, 0), 
                    text_thickness
                )
        
        # Draw ROI polygon if provided
        if roi is not None:
            output_image = roi.draw(output_image)
            
        return output_image

    def detect(self, image, roi=None):
        """
        Detect objects in the input image
        
        Args:
            image: The input image
            roi: Optional PolygonROI for filtering detections
            
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
        
        # Process output
        raw_boxes, raw_scores, raw_class_ids = self.process_output(outputs)
        
        # Apply tracking to smooth detections and reduce flickering
        # This is the key part - apply temporal smoothing and ensure we use its results
        boxes, scores, class_ids = self.tracker.update(raw_boxes, raw_scores, raw_class_ids)
        
        # Calculate inference time
        inference_time = end - start
        fps = 1 / inference_time if inference_time > 0 else 0
        
        # Draw detections ONLY using the smoothed boxes from tracker
        if boxes:  # Only draw if we have detections
            processed_image = self.draw_detections(processed_image, boxes, scores, class_ids, roi)
        else:
            # If no detections, still draw the ROI
            if roi is not None:
                processed_image = roi.draw(processed_image)
        
        # Display FPS
        cv2.putText(processed_image, f'FPS: {fps:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return boxes, scores, class_ids, processed_image
        
    def ensemble_detect(self, image, roi=None, use_flip=True):
        """
        Run detection with test-time augmentation for better accuracy
        
        Args:
            image: The input image
            roi: Optional PolygonROI for filtering detections
            use_flip: Whether to use horizontal flip augmentation
            
        Returns:
            boxes, scores, class_ids: Detection results
            processed_image: Image with detections
        """
        # Prepare input for original image
        input_tensor = self.prepare_input(image)
        
        # Perform inference on original image
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        raw_boxes, raw_scores, raw_class_ids = self.process_output(outputs)
        
        all_boxes, all_scores, all_class_ids = raw_boxes.copy(), raw_scores.copy(), raw_class_ids.copy()
        
        # Process flipped image if enabled
        if use_flip and len(raw_boxes) > 0:
            flipped = cv2.flip(image, 1)  # Horizontal flip
            flip_input = self.prepare_input(flipped)
            flip_outputs = self.session.run(self.output_names, {self.input_names[0]: flip_input})
            flip_boxes, flip_scores, flip_class_ids = self.process_output(flip_outputs)
            
            # Adjust bounding box coordinates for the flip
            for i, box in enumerate(flip_boxes):
                # Correctly flip x-coordinates (important to handle as list/array correctly)
                x1, y1, x2, y2 = box
                image_width = image.shape[1]
                flip_boxes[i] = [image_width - x2, y1, image_width - x1, y2]
            
            # Combine results
            all_boxes.extend(flip_boxes)
            all_scores.extend(flip_scores)
            all_class_ids.extend(flip_class_ids)
        
        # If no detections, handle consistently
        if not all_boxes:
            # Update tracker with empty detection to maintain consistent tracking
            smoothed_boxes, smoothed_scores, smoothed_class_ids = self.tracker.update([], [], [])
            
            # Create a copy of the image and draw ROI if needed
            processed_image = image.copy()
            if roi is not None:
                processed_image = roi.draw(processed_image)
                
            return smoothed_boxes, smoothed_scores, smoothed_class_ids, processed_image
        
        # Apply NMS on combined results
        try:
            # Convert to numpy arrays for consistent handling
            all_boxes_np = np.array(all_boxes)
            all_scores_np = np.array(all_scores)
            all_class_ids_np = np.array(all_class_ids)
            
            # Apply class-aware NMS
            final_boxes, final_scores, final_class_ids = [], [], []
            
            # Process each class separately for better NMS results
            unique_classes = np.unique(all_class_ids_np)
            for cls in unique_classes:
                cls_mask = all_class_ids_np == cls
                if np.any(cls_mask):
                    cls_boxes = all_boxes_np[cls_mask].tolist()
                    cls_scores = all_scores_np[cls_mask].tolist()
                    
                    # Apply NMS per class
                    indices = cv2.dnn.NMSBoxes(
                        cls_boxes, cls_scores, self.confidence_threshold, self.iou_threshold
                    )
                    
                    if len(indices) > 0:
                        # Handle different OpenCV versions
                        if isinstance(indices[0], np.ndarray):
                            indices = indices.flatten()
                        
                        # Add to final results
                        for idx in indices:
                            final_boxes.append(cls_boxes[idx])
                            final_scores.append(cls_scores[idx])
                            final_class_ids.append(cls)
            
            # Apply tracking to smooth detections (single source of tracking)
            smoothed_boxes, smoothed_scores, smoothed_class_ids = self.tracker.update(
                final_boxes, final_scores, final_class_ids
            )
            
            # Create processed image with smoothed boxes
            processed_image = self.draw_detections(image.copy(), smoothed_boxes, smoothed_scores, smoothed_class_ids, roi)
            
            # Display mode information
            cv2.putText(processed_image, "Ensemble Mode", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return smoothed_boxes, smoothed_scores, smoothed_class_ids, processed_image
        
        except Exception as e:
            print(f"Error during ensemble NMS: {e}")
            # Update tracker with empty detection to maintain consistent tracking
            smoothed_boxes, smoothed_scores, smoothed_class_ids = self.tracker.update([], [], [])
            
            # Create a copy of the image and draw ROI if needed
            processed_image = image.copy()
            if roi is not None:
                processed_image = roi.draw(processed_image)
                
            return smoothed_boxes, smoothed_scores, smoothed_class_ids, processed_image

class ConfidenceFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.filtered_scores = {}
    
    def filter(self, boxes, scores, class_ids):
        """Apply temporal filtering to confidence scores"""
        if not boxes:
            return boxes, scores, class_ids
        
        filtered_scores = []
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            # Create a simple hash for the box and class
            box_hash = f"{int(box[0])}-{int(box[1])}-{int(class_id)}"
            
            # Apply exponential smoothing to the score
            if box_hash in self.filtered_scores:
                smooth_score = self.alpha * score + (1 - self.alpha) * self.filtered_scores[box_hash]
            else:
                smooth_score = score
            
            # Update filtered score
            self.filtered_scores[box_hash] = smooth_score
            filtered_scores.append(smooth_score)
        
        # Clean up old entries periodically
        if len(self.filtered_scores) > 100:
            # Keep only recent entries by creating new dictionary
            # This helps prevent memory leaks
            new_dict = {}
            for box_hash in list(self.filtered_scores.keys())[-50:]:
                new_dict[box_hash] = self.filtered_scores[box_hash]
            self.filtered_scores = new_dict
        
        return boxes, filtered_scores, class_ids


# Modify the main function to support polygon ROI
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="YOLOv8 object detection on RTSP stream")
    parser.add_argument("--rtsp", type=str, required=True, help="RTSP stream URL")
    parser.add_argument("--model", type=str, default="yolov8m.onnx", help="Path to YOLOv8 ONNX model")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--record", type=str, help="Path to save video recording")
    parser.add_argument("--width", type=int, default=320, help="Model input width")
    parser.add_argument("--height", type=int, default=320, help="Model input height")
    parser.add_argument("--frame-skip", type=int, default=2, help="Process every nth frame")
    parser.add_argument("--high-accuracy", action="store_true", help="Enable ensemble mode for higher accuracy")
    parser.add_argument("--smooth", type=float, default=0.7, help="Smoothing factor for detections (0-1)")
    parser.add_argument("--roi", type=str, help="Path to polygon ROI file (will be created if not exists)")
    args = parser.parse_args()
    
    # COCO dataset labels
    labels = [
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
    
    # Initialize YOLOv8 detector with explicit input dimensions
    detector = YOLOv8Detector(
        args.model, 
        labels, 
        confidence_threshold=args.conf, 
        input_width=args.width, 
        input_height=args.height
    )
    
    # Initialize video writer if recording is enabled
    video_writer = None
    
    # Initialize polygon ROI
    roi = PolygonROI()
    roi_drawing_mode = False
    
    # Load ROI from file if provided
    if args.roi and os.path.exists(args.roi):
        if roi.load(args.roi):
            print(f"ROI loaded from {args.roi}")
        else:
            print(f"Failed to load ROI from {args.roi}, starting with empty ROI")
            roi_drawing_mode = True
    elif args.roi:
        print(f"ROI file {args.roi} not found, will create a new one")
        roi_drawing_mode = True
    
    # Open RTSP stream with threading for lower latency
    print(f"Opening RTSP stream: {args.rtsp}")
    rtsp_stream = RTSPCapture(args.rtsp)
    
    # Wait for the first frame
    first_frame = None
    max_wait = 5  # Maximum wait time in seconds
    wait_start = time.time()
    while first_frame is None:
        _, frame = rtsp_stream.read()
        if frame is not None:
            first_frame = frame
            break
        
        # Check for timeout
        if time.time() - wait_start > max_wait:
            print("Timeout waiting for first frame")
            return
        
        time.sleep(0.1)
    
    # Get video properties for recording
    if args.record:
        frame_width = first_frame.shape[1]
        frame_height = first_frame.shape[0]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(args.record, fourcc, 20, (frame_width, frame_height))
        print(f"Recording to {args.record}")
    
    # Mouse callback function for ROI drawing
    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_drawing_mode
        
        if not roi_drawing_mode:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            roi.add_point(x, y)
            print(f"Added point ({x}, {y}) to ROI")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if roi.complete_polygon():
                roi_drawing_mode = False
                print("ROI polygon completed")
                # Save ROI if filename provided
                if args.roi:
                    roi.save(args.roi)
    
    # Create window and set mouse callback
    window_name = "YOLOv8 Object Detection"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\nControls:")
    print("  Press 'q' to exit")
    print("  Press 'a' to toggle high-accuracy mode")
    print("  Press 's' to save a screenshot")
    print("  Press 'r' to reset/redraw ROI")
    print("  Press 'f' to finish drawing ROI")
    print("  Left-click to add points to the ROI\n")
        
    if roi_drawing_mode:
        print("ROI drawing mode active - click to add points, right-click to complete")
    
    # Variables for FPS calculation
    frame_count = 0
    fps = 0
    fps_start_time = time.time()
    processing_times = deque(maxlen=30)  # Keep track of recent processing times
    
    # Main processing loop
    high_accuracy_mode = args.high_accuracy
    frame_skip_counter = 0
    last_processed_frame = None  # Store the last processed frame
    
    try:
        while True:
            # Read frame from RTSP stream
            ret, frame = rtsp_stream.read()
            
            # If frame is not available, try again
            if not ret or frame is None:
                print("Failed to receive frame, retrying...")
                time.sleep(0.1)
                continue
            
            # Calculate display FPS
            frame_count += 1
            if frame_count >= 10:
                end_time = time.time()
                fps = frame_count / (end_time - fps_start_time)
                fps_start_time = end_time
                frame_count = 0
            
            # If in ROI drawing mode, just show the frame with current ROI
            if roi_drawing_mode:
                cv2.putText(
                    display_frame,
                    "ROI Drawing Mode: Click to add points, press 'F' to complete",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                
                # Display frame and save if recording
                cv2.imshow(window_name, display_frame)
                if video_writer is not None:
                    video_writer.write(display_frame)
            
            # Process frame for object detection if not in drawing mode
            else:
                # Process frame or use the last processed frame
                if frame_skip_counter % args.frame_skip == 0:
                    # Process this frame
                    process_start = time.time()
                    
                    # Detect objects with ROI filtering
                    if high_accuracy_mode:
                        boxes, scores, class_ids, processed_frame = detector.ensemble_detect(frame, roi)
                    else:
                        boxes, scores, class_ids, processed_frame = detector.detect(frame, roi)
                    
                    # Store processed frame
                    last_processed_frame = processed_frame.copy()
                    
                    # Calculate processing time
                    process_end = time.time()
                    process_time = process_end - process_start
                    processing_times.append(process_time)
                    
                    # Add FPS information
                    if len(processing_times) > 0:
                        avg_process_time = sum(processing_times) / len(processing_times)
                        process_fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
                        
                        cv2.putText(
                            last_processed_frame,
                            f"Process FPS: {process_fps:.1f}",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                    
                    # Add display FPS info
                    cv2.putText(
                        last_processed_frame,
                        f"Display FPS: {fps:.1f}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    # Add accuracy mode indicator
                    cv2.putText(
                        last_processed_frame,
                        f"Mode: {'High Accuracy' if high_accuracy_mode else 'Standard'}",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    # Add ROI status indicator
                    cv2.putText(
                        last_processed_frame,
                        f"ROI: {'Active' if roi.is_complete else 'Inactive'}",
                        (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                else:
                    # For skipped frames, we still need to update FPS info
                    if last_processed_frame is not None:
                        # Create a copy to avoid modifying the original
                        display_frame = last_processed_frame.copy()
                        
                        # Update FPS counters on the display frame
                        if len(processing_times) > 0:
                            avg_process_time = sum(processing_times) / len(processing_times)
                            process_fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
                            
                            cv2.putText(
                                display_frame,
                                f"Process FPS: {process_fps:.1f}",
                                (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2
                            )
                        
                        cv2.putText(
                            display_frame,
                            f"Display FPS: {fps:.1f}",
                            (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                        
                        cv2.putText(
                            display_frame,
                            f"Mode: {'High Accuracy' if high_accuracy_mode else 'Standard'}",
                            (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                        
                        cv2.putText(
                            display_frame,
                            f"ROI: {'Active' if roi.is_complete else 'Inactive'}",
                            (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                    else:
                        # If no frame has been processed yet, use the original frame
                        display_frame = frame.copy()
                        roi.draw(display_frame)  # Draw ROI on original frame
                
                # Increment frame skip counter
                frame_skip_counter += 1
                if frame_skip_counter >= 1000:  # Reset to prevent overflow
                    frame_skip_counter = 0
                
                # Display the frame
                frame_to_show = last_processed_frame if last_processed_frame is not None else display_frame
                cv2.imshow(window_name, frame_to_show)
                
                # Save frame if recording
                if video_writer is not None:
                    video_writer.write(frame_to_show)
            
            # Check for key presses
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('a'):
                high_accuracy_mode = not high_accuracy_mode
                print(f"High accuracy mode: {'ON' if high_accuracy_mode else 'OFF'}")
                # Reset frame counter to force processing on next frame
                frame_skip_counter = 0
            elif key == ord('s'):
                # Save screenshot
                screenshot_path = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(screenshot_path, frame_to_show)
                print(f"Screenshot saved to {screenshot_path}")
            elif key == ord('r'):
                # Reset ROI and enable drawing mode
                roi = PolygonROI()
                roi_drawing_mode = True
                print("ROI reset - drawing mode activated")
            elif key in [ord('f'), ord('F')]:
                if roi_drawing_mode:
                    if roi.complete_polygon():
                        roi_drawing_mode = False
                        print("ROI polygon completed")
                        if args.roi:
                            roi.save(args.roi)
                    else:
                        print("Need at least 3 points to complete the polygon")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Release resources
        rtsp_stream.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print("Resources released")

if __name__ == "__main__":
    main()