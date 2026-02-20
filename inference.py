import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
import cv2

TFLITE_MODEL_PATH = "exported_model/road_anomaly_detector.tflite"
IMG_SIZE = (416, 416)
CLASSES = ["anomaly"]

IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5

interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero_point = input_details[0]['quantization']
output_scales = [d['quantization'][0] for d in output_details if d['quantization'][0] != 0]
output_zero_points = [d['quantization'][1] for d in output_details if d['quantization'][0] != 0]

def yolo_boxes(predictions, anchors, num_classes):
    b, g, g2, a, _ = predictions.shape
    assert g == g2
    anchors = tf.constant(anchors, dtype=tf.float32)
    box_xy = tf.sigmoid(predictions[..., :2])
    objectness = tf.sigmoid(predictions[..., 4])
    grid = tf.meshgrid(tf.range(g), tf.range(g))
    grid = tf.stack(grid, axis=-1)
    grid = tf.expand_dims(grid, axis=2)
    grid = tf.tile(grid, [1, 1, a, 1])
    grid = tf.cast(grid, tf.float32)
    box_xy = (box_xy + grid) / tf.cast(g, tf.float32)
    box_wh = tf.exp(predictions[..., 2:4]) * anchors[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
    bbox = tf.reshape(bbox, (b, g * g * a, 4))
    probabilities = tf.sigmoid(predictions[..., 5:]) * tf.expand_dims(objectness, -1)
    probabilities = tf.reshape(probabilities, (b, g * g * a, num_classes))
    return bbox, probabilities

def nms(boxes, scores, iou_threshold):
    selected_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=100, iou_threshold=iou_threshold)
    return selected_indices

ANCHORS = [[[10,13], [16,30], [33,23]],
           [[30,61], [62,45], [59,119]],
           [[116,90], [156,198], [373,326]]]

def run_inference_on_frame(frame):
    original_shape = frame.shape[:2]
    img = cv2.resize(frame, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    input_data = np.expand_dims(img, axis=0)
    input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    outputs = []
    for i, detail in enumerate(output_details):
        output = interpreter.get_tensor(detail['index'])
        scale, zero_point = output_scales[i], output_zero_points[i]
        output = (output.astype(np.float32) - zero_point) * scale
        outputs.append(output)
    
    all_boxes = []
    all_scores = []
    all_classes = []
    for i, output in enumerate(outputs):
        boxes, probs = yolo_boxes(output, ANCHORS[i], NUM_CLASSES)
        scores = tf.reduce_max(probs, axis=-1)
        classes = tf.argmax(probs, axis=-1)
        mask = scores > SCORE_THRESHOLD
        boxes = tf.boolean_mask(boxes, mask)
        scores = tf.boolean_mask(scores, mask)
        classes = tf.boolean_mask(classes, mask)
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_classes.append(classes)
    
    all_boxes = tf.concat(all_boxes, axis=0)
    all_scores = tf.concat(all_scores, axis=0)
    all_classes = tf.concat(all_classes, axis=0)
    
    selected_indices = nms(all_boxes, all_scores, IOU_THRESHOLD)
    selected_boxes = tf.gather(all_boxes, selected_indices)
    selected_scores = tf.gather(all_scores, selected_indices)
    selected_classes = tf.gather(all_classes, selected_indices)
    
    selected_boxes *= np.array([original_shape[1], original_shape[0], original_shape[1], original_shape[0]])
    
    for box, score, cls in zip(selected_boxes, selected_scores, selected_classes):
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f"{CLASSES[cls]}: {score:.2f}"
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 0 for default camera (use picamera module if needed for Pi Camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set to ~360p width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)  # Set to 360p height
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = run_inference_on_frame(frame)
        cv2.imshow('Live Anomaly Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
