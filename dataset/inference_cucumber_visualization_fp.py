import numpy as np
import os
import sys
import tensorflow as tf
import datetime
import functools

#from matplotlib import pyplot as plt
import cv2
from PIL import Image

sys.path.append(".")
import visualization_with_eval

sys.path.append("/home/maria/TFM/models/research/slim")
sys.path.append("/home/maria/TFM/models/research")
from object_detection.utils import ops as utils_ops
from object_detection import eval_util
from object_detection import evaluator as evtor
from object_detection.builders import model_builder
from object_detection.utils import dataset_util
from object_detection.builders import dataset_builder

sys.path.append("/home/maria/TFM/models/research/object_detection")
from utils import label_map_util
from utils import visualization_utils as vis_util
from utils import object_detection_evaluation
from utils import config_util

MAX_DIMENSION = 780
PIPELINE_CONFIG_PATH = '/home/maria/TFM/cucumbers-git/models/faster_rcnn_resnet101_pets.config'
PATH_TO_CKPT = '/home/maria/TFM/cucumbers-git/models/exported_graphs_exp3/frozen_inference_graph.pb'
PATH_TO_LABELS = 'cucumber_label_map.pbtxt'
NUM_CLASSES = 1
PATH_TO_TEST_IMAGES_DIR = 'images/test_new_set'

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def resize_img (image):
    h, w = image.shape[:2]
    r = MAX_DIMENSION / w
    new_h = int(h * r)
    img = cv2.resize(image, (MAX_DIMENSION, new_h))
    return img

def run_inference_for_single_image(image, graph, evaluator):
  with graph.as_default():
    with tf.Session() as sess:
      configs = config_util.get_configs_from_pipeline_file(
        PIPELINE_CONFIG_PATH)

      def get_next(config):
        return dataset_util.make_initializable_iterator(
            dataset_builder.build(config)).get_next()
      create_input_dict_fn = functools.partial(get_next, configs['eval_input_config'])

      eval_config = configs['eval_config']
      model_fn = functools.partial(
        model_builder.build,
        model_config=configs['model'],
        is_training=False)
      model = model_fn()
      tensor_dict = evtor._extract_prediction_tensors(
        model=model,
        create_input_dict_fn=create_input_dict_fn,
        ignore_groundtruth=eval_config.ignore_groundtruth)


      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      #tensor_dict = {}
      # for key in [
      #     'num_detections', 'detection_boxes', 'detection_scores',
      #     'detection_classes', 'detection_masks'
      # ]:
      #   tensor_name = key + ':0'
      #   if tensor_name in all_tensor_names:
      #     tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
      #         tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

      evaluator.add_single_ground_truth_image_info(
          image_id=1, groundtruth_dict=output_dict)
      evaluator.add_single_detected_image_info(
          image_id=1, detections_dict=output_dict)
      metrics = evaluator.evaluate()
      evaluator.clear()
  return output_dict

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

test_image_paths = os.listdir(PATH_TO_TEST_IMAGES_DIR)

for image_path in test_image_paths:
  image = Image.open(os.path.join(PATH_TO_TEST_IMAGES_DIR,image_path))
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  image_np = resize_img(image_np)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  evaluator = object_detection_evaluation.PascalDetectionEvaluator(categories=categories)
  output_dict = run_inference_for_single_image(image_np, detection_graph, evaluator)

  # Visualization of the results of a detection.
  cv2.imshow('cucumbers test', image_np)
  cv2.waitKey()

  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  #cv2.namedWindow('cucumbers test', cv2.WINDOW_NORMAL)
  # cv2.resizeWindow('cucumbers test', IMAGE_SIZE[0], IMAGE_SIZE[1])
  cv2.imshow('cucumbers test', image_np)
  key = cv2.waitKey()
  if key == 121:
      cv2.imwrite(datetime.datetime.now().isoformat()+'.jpg', image_np)
  cv2.destroyWindow('cucumbers test')

