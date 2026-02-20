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
