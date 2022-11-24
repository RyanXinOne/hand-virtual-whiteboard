import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential


class Convolutional(layers.Layer):
    """
    Conv2d + BatchNorm + LeakyReLU
    """

    def __init__(self, filters, kernel_size, strides, batch_normalize, activation):
        super(Convolutional, self).__init__()
        self.pipeline = Sequential()
        self.pipeline.add(layers.Conv2D(filters, kernel_size, strides, padding='same', use_bias=not batch_normalize))
        if batch_normalize:
            self.pipeline.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
        if activation == 'leaky':
            self.pipeline.add(layers.LeakyReLU(alpha=0.1))

    def call(self, inputs, training=False):
        return self.pipeline(inputs, training=training)


class Upsample(layers.Layer):
    """
    Double upsampling
    """
    stride = 2

    def __init__(self):
        super(Upsample, self).__init__()
        self.upsample = layers.UpSampling2D(size=self.stride, interpolation='nearest')

    def call(self, inputs):
        return self.upsample(inputs)


class Shortcut(layers.Layer):
    "ResNet block"

    def __init__(self,
                 filters1, kernel_size1, strides1, batch_normalize1, activation1,
                 filters2, kernel_size2, strides2, batch_normalize2, activation2):
        super(Shortcut, self).__init__()
        self.conv1 = Convolutional(filters1, kernel_size1, strides1, batch_normalize1, activation1)
        self.conv2 = Convolutional(filters2, kernel_size2, strides2, batch_normalize2, activation2)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        return x + inputs


class Route(layers.Layer):
    "Customised route for concatenation"

    def __init__(self):
        super(Route, self).__init__()

    def call(self, inputs_list):
        return tf.concat(inputs_list, axis=-1)


class Yolo(layers.Layer):
    """
    Yolo output layer
    """
    anchors_dict = {
        "small": [(10, 13), (16, 30), (33, 23)],
        "medium": [(30, 61), (62, 45), (59, 119)],
        "large": [(116, 90), (156, 198), (373, 326)]
    }
    ignore_thres = 0.5
    mse_loss = tf.keras.losses.MeanSquaredError()
    bce_loss = tf.keras.losses.BinaryCrossentropy()
    obj_scale = 1
    noobj_scale = 100

    def __init__(self, anchors_type, num_classes, img_dim):
        super(Yolo, self).__init__()
        self.anchors = tf.constant(self.anchors_dict[anchors_type], dtype=tf.float32)
        self.num_anchor = self.anchors.shape[0]
        self.num_classes = num_classes
        self.img_dim = img_dim  # height of raw image

    def build(self, input_shape):
        self.num_samples = input_shape[0]  # batch size
        self.grid_size = input_shape[2]  # height of the input feature map
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = tf.expand_dims(tf.expand_dims(tf.repeat(tf.expand_dims(tf.range(self.grid_size, dtype=tf.float32), 0), self.grid_size, 0), 0), 0)
        self.grid_y = tf.expand_dims(tf.expand_dims(tf.repeat(tf.expand_dims(tf.range(self.grid_size, dtype=tf.float32), 1), self.grid_size, 1), 0), 0)
        self.scaled_anchors = self.anchors / self.stride
        self.anchor_w = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.scaled_anchors[:, 0], 1), 1), 0)
        self.anchor_h = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.scaled_anchors[:, 1], 1), 1), 0)

    def call(self, inputs, targets=None):
        prediction = tf.transpose(
            tf.reshape(inputs, (self.num_samples, self.num_anchor, self.num_classes + 5, self.grid_size, self.grid_size)),
            perm=(0, 1, 3, 4, 2)
        )

        # Get outputs
        x = tf.math.sigmoid(prediction[..., 0])  # Center x
        y = tf.math.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = tf.math.sigmoid(prediction[..., 4])  # Conf
        pred_cls = tf.math.sigmoid(prediction[..., 5:])  # Cls pred.

        # Add offset and scale with anchors
        pred_boxes = tf.stack([
            x + self.grid_x,
            y + self.grid_y,
            tf.math.exp(w) * self.anchor_w,
            tf.math.exp(h) * self.anchor_h
        ], axis=-1)

        output = tf.concat([
            tf.reshape(pred_boxes, (self.num_samples, -1, 4)) * self.stride,
            tf.reshape(pred_conf, (self.num_samples, -1, 1)),
            tf.reshape(pred_cls, (self.num_samples, -1, self.num_classes))
        ], axis=-1)

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = self.build_targets(pred_boxes, pred_cls, targets, self.scaled_anchors, self.ignore_thres)
            pass

    def build_targets(self, pred_boxes, pred_cls, target, anchors, ignore_thres):
        nB = pred_boxes.shape[0]
        nA = pred_boxes.shape[1]
        nC = pred_cls.shape[-1]
        nG = pred_boxes.shape[2]

        # Output tensors
        obj_mask = np.zeros((nB, nA, nG, nG), dtype=np.int8)
        noobj_mask = np.ones((nB, nA, nG, nG), dtype=np.int8)
        class_mask = np.zeros((nB, nA, nG, nG), dtype=np.float32)
        iou_scores = np.zeros((nB, nA, nG, nG), dtype=np.float32)
        tx = np.zeros((nB, nA, nG, nG), dtype=np.float32)
        ty = np.zeros((nB, nA, nG, nG), dtype=np.float32)
        tw = np.zeros((nB, nA, nG, nG), dtype=np.float32)
        th = np.zeros((nB, nA, nG, nG), dtype=np.float32)
        tcls = np.zeros((nB, nA, nG, nG, nC), dtype=np.float32)

        # Convert to position relative to box
        target_boxes = target[:, 2:6] * nG
        gxy = target_boxes[:, :2]
        gwh = target_boxes[:, 2:]
        # Get anchors with best iou
        ious = tf.stack([self.bbox_wh_iou(anchor, gwh) for anchor in anchors])
        best_ious, best_n = tf.math.reduce_max(ious, axis=0), tf.math.argmax(ious, axis=0)
        # Separate target values
        b, target_labels = tf.transpose(tf.cast(target[:, :2], tf.int32))
        gx, gy = tf.transpose(gxy)
        gw, gh = tf.transpose(gwh)
        gi, gj = tf.transpose(tf.cast(gxy, tf.int32))
        # Set masks
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0

        # Set noobj mask to zero where iou exceeds ignore threshold
        for i, anchor_ious in enumerate(tf.transpose(ious)):
            noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

        # Coordinates
        tx[b, best_n, gj, gi] = gx - tf.math.floor(gx)
        ty[b, best_n, gj, gi] = gy - tf.math.floor(gy)
        # Width and height
        tw[b, best_n, gj, gi] = tf.math.log(gw / tf.gather(anchors, best_n)[:, 0] + 1e-16)
        th[b, best_n, gj, gi] = tf.math.log(gh / tf.gather(anchors, best_n)[:, 1] + 1e-16)
        # One-hot encoding of label
        tcls[b, best_n, gj, gi, target_labels] = 1
        # Compute label correctness and iou at best anchor
        class_mask[b, best_n, gj, gi] = tf.cast(tf.argmax(pred_cls[b, best_n, gj, gi], -1) == target_labels, tf.float32)
        iou_scores[b, best_n, gj, gi] = self.bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

        tconf = obj_mask
        return tf.constant(iou_scores), tf.constant(class_mask), tf.constant(obj_mask), tf.constant(noobj_mask), tf.constant(tx), tf.constant(ty), tf.constant(tw), tf.constant(th), tf.constant(tcls), tf.constant(tconf, dtype=tf.float32)

    def bbox_wh_iou(self, wh1, wh2):
        wh2 = tf.transpose(wh2)
        w1, h1 = wh1[0], wh1[1]
        w2, h2 = wh2[0], wh2[1]
        inter_area = tf.math.minimum(w1, w2) * tf.math.minimum(h1, h2)
        union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
        return inter_area / union_area

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        Returns the IoU of two bounding boxes
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # get the coordinates of the intersection rectangle
        inter_rect_x1 = tf.math.maximum(b1_x1, b2_x1)
        inter_rect_y1 = tf.math.maximum(b1_y1, b2_y1)
        inter_rect_x2 = tf.math.minimum(b1_x2, b2_x2)
        inter_rect_y2 = tf.math.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = tf.math.maximum(inter_rect_x2 - inter_rect_x1 + 1, 0) * tf.math.maximum(inter_rect_y2 - inter_rect_y1 + 1, 0)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou


class Darknet(Model):
    def __init__(self):
        super(Darknet, self).__init__()
        pass

    def call(self, inputs):
        pass


if __name__ == "__main__":
    yolo = Yolo("large", 80, 448)
    x = tf.random.normal((4, 255, 13, 13))
    targets = tf.constant([[0.0000e+00, 0.0000e+00, 4.5199e-01, 5.6075e-01, 2.5798e-01, 3.7675e-01],
                           [0.0000e+00, 3.5000e+01, 5.5551e-01, 5.0212e-01, 5.7078e-02, 5.3797e-02],
                           [1.0000e+00, 1.4000e+01, 4.2912e-01, 5.3070e-01, 6.5458e-01, 6.7075e-01],
                           [1.0000e+00, 7.3000e+01, 7.6035e-01, 4.9838e-01, 5.8172e-02, 1.9622e-01],
                           [1.0000e+00, 5.8000e+01, 5.9236e-01, 2.3907e-01, 3.4219e-01, 1.6889e-01],
                           [1.0000e+00, 7.3000e+01, 7.0812e-01, 4.5338e-01, 5.1469e-02, 1.0936e-01],
                           [2.0000e+00, 1.3000e+01, 2.3761e-01, 7.5487e-01, 4.7228e-01, 1.3944e-01],
                           [3.0000e+00, 4.0000e+01, 2.9433e-01, 5.5127e-01, 2.1300e-01, 5.7480e-01],
                           [3.0000e+00, 4.0000e+01, 2.0783e-01, 5.2943e-01, 1.6566e-01, 4.5017e-01],
                           [3.0000e+00, 4.0000e+01, 6.5477e-01, 5.4831e-01, 3.9552e-01, 8.6741e-01],
                           [3.0000e+00, 4.0000e+01, 5.5523e-01, 5.3773e-01, 1.7722e-01, 5.1236e-01],
                           [3.0000e+00, 6.0000e+01, 7.6777e-01, 8.3040e-01, 1.9639e-01, 3.0473e-01],
                           [3.0000e+00, 6.0000e+01, 3.8280e-01, 8.2727e-01, 5.1169e-01, 3.1430e-01]])
    yolo(x, targets)
