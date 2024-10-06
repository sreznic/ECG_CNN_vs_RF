import tensorflow as tf

def weighted_categorical_crossentropy(target, output, weights, axis=-1):
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)
    target.shape.assert_is_compatible_with(output.shape)
    weights = tf.reshape(tf.convert_to_tensor(weights, dtype=target.dtype), (1,-1))

    # Adjust the predictions so that the probability of
    # each class for every sample adds up to 1
    # This is needed to ensure that the cross entropy is
    # computed correctly.
    output = output / tf.reduce_sum(output, axis, True)

    # Compute cross entropy from probabilities.
    epsilon_ = tf.constant(tf.keras.backend.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    return -tf.reduce_sum(weights * target * tf.math.log(output), axis=axis)

class WeightedCategoricalCrossentropy:
    def __init__(
        self,
        weights,
        label_smoothing=0.0,
        axis=-1,
        name="weighted_categorical_crossentropy",
        fn = None,
    ):
        """Initializes `WeightedCategoricalCrossentropy` instance.
        Args:
          from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
            assume that `y_pred` contains probabilities (i.e., values in [0,
            1]).
          label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When >
            0, we compute the loss between the predicted labels and a smoothed
            version of the true labels, where the smoothing squeezes the labels
            towards 0.5.  Larger values of `label_smoothing` correspond to
            heavier smoothing.
          axis: The axis along which to compute crossentropy (the features
            axis).  Defaults to -1.
          name: Name for the op. Defaults to 'weighted_categorical_crossentropy'.
        """
        super().__init__()
        self.weights = weights # tf.reshape(tf.convert_to_tensor(weights),(1,-1))
        self.label_smoothing = label_smoothing
        self.name = name
        self.fn = weighted_categorical_crossentropy if fn is None else fn

    def __call__(self, y_true, y_pred, axis=-1):
        if isinstance(axis, bool):
            raise ValueError(
                "`axis` must be of type `int`. "
                f"Received: axis={axis} of type {type(axis)}"
            )
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        self.label_smoothing = tf.convert_to_tensor(self.label_smoothing, dtype=y_pred.dtype)

        if y_pred.shape[-1] == 1:
            warnings.warn(
                "In loss categorical_crossentropy, expected "
                "y_pred.shape to be (batch_size, num_classes) "
                f"with num_classes > 1. Received: y_pred.shape={y_pred.shape}. "
                "Consider using 'binary_crossentropy' if you only have 2 classes.",
                SyntaxWarning,
                stacklevel=2,
            )

        def _smooth_labels():
            num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
            return y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)

        y_true = tf.__internal__.smart_cond.smart_cond(self.label_smoothing, _smooth_labels, lambda: y_true)

        return tf.reduce_mean(self.fn(y_true, y_pred, self.weights, axis=axis))
    
    def get_config(self):
        config = {"name":self.name, "weights": self.weights, "fn": weighted_categorical_crossentropy}

        # base_config = super().get_config()
        return dict(list(config.items()))