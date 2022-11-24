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
            pass


class Darknet(Model):
    def __init__(self):
        super(Darknet, self).__init__()
        pass

    def call(self, inputs):
        pass


if __name__ == "__main__":
    yolo = Yolo("large", 80, 448)
    x = tf.random.normal((4, 255, 14, 14))
    yolo(x)
