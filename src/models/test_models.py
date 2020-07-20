# Test images with pretrained tensorflow models
# call :
# python models/research/object_detection/test_models.py --frozen_graph {$path_to_frozen_graph}
# --label_map {$path_to_##.pbtxt} --test_dir {$path_to_test_set}
# --num {$number_test_images} --output_dir {$path_to_save_output_imagesWITHboxes}

import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image

# imports from object detection module
# From tensorflow/models/research/
# protoc object_detection/protos/*.proto --python_out=.
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

parser = argparse.ArgumentParser(description="Load tensorflow model and do object detection inference ")
parser.add_argument("--frozen_graph", type=str,
                    default="models/faster_rcnn_resnet50/train20190612131800test/frozen_inference_graph.pb",
                    help="the path to the frozen graph")
parser.add_argument("--label_map", type=str, default="data/processed/label_map.pbtxt",
                    help="the path to the label map")
parser.add_argument("--num", type=int, default=3, help="the number of test images, set negative to test"
                                                       "all images in the test_dir. ")
parser.add_argument("--output_dir", type=str, default="data/processed/tested",
                    help="the directory to store the output of object inference")
parser.add_argument("--seedling", action="store_true", default=False,
                    help="if only the images with seedlings are applied")

args = parser.parse_args()


def load_label_map(label_path):
    """
    Label maps map indices to category names, so that when our convolution network predicts 5,
     we know that this corresponds to airplane. Here we use internal utility functions,
    but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    :param label_path: the path to the label map
    :return: cateogry index from label map
    """
    return label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)


def load_image_into_numpy_array(image):
    # helper corde
    (im_width, im_height) = image.size
    print('loaded image: ', image.size)
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_test_image_paths(seedling, num):
    all_images = []
    if seedling:
        with open('data/interim/datasets/test_seedling_paths')as file:
            for line in file.readlines():
                name = line.replace('\n', '')
                all_images.append(name)
    else:
        with open('data/interim/datasets/test_paths')as file:
            for line in file.readlines():
                name = line.replace('\n', '')
                all_images.append(name)

    if num >= 0:
        return all_images[:num]
    else:
        return all_images


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

                # Reframe is required to translate mask from box coordinates to image coordinates
                # and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

        return output_dict


def main():
    # load a frozen tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # loading label map
    category_index = load_label_map(args.label_map)

    TEST_IMAGE_PATHS = get_test_image_paths(args.seedling, args.num)

    for image_path in TEST_IMAGE_PATHS:
        image_name = str(image_path).split('/')[-1]
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)

        # Visualization of the results of a detection.
        # uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
        boxed_image = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)

        # save path for images with overlaid boxes
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        saved_path = os.path.join(args.output_dir, image_name)

        # convert numpy array to image
        im = Image.fromarray(boxed_image)
        # save image
        im.save(saved_path)
        print('saved image to {}'.format(saved_path))


if __name__ == '__main__':
    main()
