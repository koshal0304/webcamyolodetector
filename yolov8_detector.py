import cv2
import numpy as np
import onnxruntime as ort
import time
import os

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

def main():
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
    
    # YOLOv8 ONNX model path (use the one you just exported)
    model_path = "yolov8n.onnx"  # Update this path if your model is in a different location
    
    # Initialize YOLOv8 detector with explicit input dimensions
    # Use lower confidence threshold for testing
    detector = YOLOv8Detector(model_path, labels, confidence_threshold=0.3, input_width=640, input_height=640)
    
    # Open webcam (0) or video file
    # For video file: cap = cv2.VideoCapture("path/to/your/video.mp4")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to exit.")
    
    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        try:
            # Detect objects
            boxes, scores, class_ids, processed_frame = detector.detect(frame)
            
            # Display output
            cv2.imshow("YOLOv8 Detection", processed_frame)
        except Exception as e:
            print(f"Error during detection: {e}")
            # Display original frame if detection fails
            cv2.imshow("YOLOv8 Detection", frame)
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Detection finished.")

# Function to process a single image
def process_image(image_path):
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
    
    # YOLOv8 ONNX model path
    model_path = "yolov8n.onnx"  # Update this path if needed
    
    # Initialize YOLOv8 detector with explicit input dimensions
    # Use lower confidence threshold for testing
    detector = YOLOv8Detector(model_path, labels, confidence_threshold=0.3, input_width=640, input_height=640)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    try:
        # Detect objects
        boxes, scores, class_ids, processed_image = detector.detect(image)
        
        # Display output
        cv2.imshow("YOLOv8 Detection", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Print detection results
        print(f"Found {len(boxes)} objects:")
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = [int(round(coord)) for coord in box]
            class_name = labels[class_id] if 0 <= class_id < len(labels) else f"Unknown ({class_id})"
            print(f"{i+1}. {class_name}: {score:.2f} at [{x1}, {y1}, {x2}, {y2}]")
        
        # Save output image
        output_path = "output_" + os.path.basename(image_path)
        cv2.imwrite(output_path, processed_image)
        print(f"Saved output image to {output_path}")
    
    except Exception as e:
        print(f"Error during detection: {e}")

if __name__ == "__main__":
    # Run webcam detection
    main()
    
    # Or process a single image (uncomment to use)
    # process_image("path/to/your/image.jpg")