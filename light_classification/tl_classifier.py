import tensorflow as tf
import os
import numpy as np

class TLClassifier(object):
    def __init__(self,
                 model_path,
                 img_height,
                 img_width):
        self.model_path = model_path
        self.img_height = img_height
        self.img_width = img_width

        # load frozen graph
        self.graph = None
        self.session = None
        self.image_tensor = None
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_classes = None
        self.load_model(model_path=self.model_path)

        # threshold for detection
        self.threshold_prob = 0.7

    def load_model(self, model_path):
        # load serialized Tensor Graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

        self.session = tf.Session(graph=self.graph)

        # update tensor
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')

        # run first prediction with dummy image to initialize (since first run is considerably slow)
        self.warmup_model()

    def warmup_model(self):
        # create a null image
        null_img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        # warm up tensorflow session by running with a null-image
        _ = self.session.run(
            [self.detection_boxes, self.detection_classes, self.detection_scores],
            feed_dict={
                self.image_tensor: np.expand_dims(null_img, axis=0)
            }
        )

    def get_classification(self, image_rgb):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        # run the detection
        _, classes, scores = self.get_detections(image_rgb)


        # we get the one give maximum score
        i_max_score = np.argmax(scores)
        if (scores[i_max_score] > self.threshold_prob):
            return classes[i_max_score]

        return 4

    def get_detections(self, image_rgb):
        (boxes, classes, scores) = self.session.run(
            [self.detection_boxes,
             self.detection_classes,
             self.detection_scores],
            feed_dict={
                self.image_tensor: np.expand_dims(image_rgb, axis=0)
            }
        )

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        return boxes, classes, scores