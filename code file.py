import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

IMAGE_DIR = "dataset/images"
ANNOTATION_DIR = "dataset/annotations"
LABEL_MAP = "dataset/label_map.txt"
TFRECORD_DIR = "data"
TFLITE_MODEL_PATH = "exported_model/road_anomaly_detector.tflite"
BATCH_SIZE = 4
EPOCHS = 10
IMG_SIZE = (416, 416)
NUM_CLASSES = 1

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(ANNOTATION_DIR, exist_ok=True)
os.makedirs(TFRECORD_DIR, exist_ok=True)

with open(LABEL_MAP, 'w') as f:
    f.write("anomaly\n")

with open('data/anomaly.names', 'w') as f:
    f.write("anomaly\n")

for i in range(5):
    img = np.random.uniform(0, 1, size=(IMG_SIZE[0], IMG_SIZE[1], 3)) * 255
    img = Image.fromarray(img.astype(np.uint8))
    img_path = os.path.join(IMAGE_DIR, f"img{i}.jpg")
    img.save(img_path)
    
    if i % 2 == 0:
        with open(os.path.join(ANNOTATION_DIR, f"img{i}.txt"), 'w') as f:
            f.write("0 0.5 0.5 0.5 0.5\n")
            f.write("0 0.3 0.3 0.2 0.2\n")

def yolo_to_csv(image_dir, annotation_dir, output_csv):
    data = []
    for img_file in os.listdir(image_dir):
        if not img_file.endswith((".jpg", ".png")):
            continue
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path)
        width, height = img.size
        ann_file = os.path.join(annotation_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))
        if os.path.exists(ann_file):
            with open(ann_file, "r") as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    class_id = int(parts[0])
                    x_center, y_center, w, h = parts[1:]
                    xmin = int((x_center - w / 2) * width)
                    xmax = int((x_center + w / 2) * width)
                    ymin = int((y_center - h / 2) * height)
                    ymax = int((y_center + h / 2) * height)
                    class_name = "anomaly"
                    data.append([img_file, width, height, class_name, xmin, ymin, xmax, ymax])
    df = pd.DataFrame(data, columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    df.to_csv(output_csv, index=None)
    print(f"CSV saved to {output_csv}")

yolo_to_csv(IMAGE_DIR, ANNOTATION_DIR, 'data/train.csv')

os.system('cp data/train.csv data/test.csv')

def class_text_to_int(row_label):
    if row_label == 'anomaly':
        return 1
    else:
        return 0

def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename.iloc[0])), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = tf.io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.iloc[0].encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))
    return tf_example

def main(csv_input, output_path, image_dir):
    writer = tf.io.TFRecordWriter(output_path)
    path = image_dir
    examples = pd.read_csv(csv_input)
    grouped = examples.groupby('filename')
    for name, group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))

main('data/train.csv', 'data/train.tfrecord', IMAGE_DIR)
main('data/test.csv', 'data/test.tfrecord', IMAGE_DIR)

os.system('wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights')
os.system('python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf')

os.system(f'python train.py \
  --dataset ./data/train.tfrecord \
  --val_dataset ./data/test.tfrecord \
  --classes ./data/anomaly.names \
  --num_classes {NUM_CLASSES} \
  --mode fit --transfer darknet \
  --batch_size {BATCH_SIZE} \
  --epochs {EPOCHS} \
  --weights ./checkpoints/yolov3.tf \
  --weights_num_classes 80')

os.system('python export_tfserving.py --checkpoint ./checkpoints/yolov3_train_{EPOCHS}.tf --output ./serving/1')

def representative_dataset():
    dataset = tf.data.TFRecordDataset('data/train.tfrecord')
    for raw_record in dataset.take(100):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        img = tf.image.decode_jpeg(example.features.feature['image/encoded'].bytes_list.value[0])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, [416, 416])
        yield [tf.expand_dims(img, 0)]

converter = tf.lite.TFLiteConverter.from_saved_model('./serving/1')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

os.makedirs(os.path.dirname(TFLITE_MODEL_PATH), exist_ok=True)
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)
print(f"TFLite model saved at {TFLITE_MODEL_PATH}")
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
