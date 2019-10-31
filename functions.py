import tensorflow as tf

def expand_moments_dim(moment):
    return tf.reshape(moment, [-1, 1, 1, tf.shape(moment)[-1]])

def adain(content_feature, style_feature, eps=1e-5):
    content_mean, content_std = tf.nn.moments(content_feature, axes=[1, 2])
    style_mean, style_std = tf.nn.moments(style_feature, axes=[1, 2])

    content_mean = expand_moments_dim(content_mean)
    # TFLite does not support broadcasting; it is allowed for add, mul, sub, div
    # content_mean = tf.broadcast_to(content_mean, tf.shape(content_feature))

    content_std = expand_moments_dim(content_std)+eps
    # TFLite does not support broadcasting; it is allowed for add, mul, sub, div
    # content_std = tf.broadcast_to(content_std, tf.shape(content_feature))

    style_mean = expand_moments_dim(style_mean)
    # TFLite does not support broadcasting; it is allowed for add, mul, sub, div
    # style_mean = tf.broadcast_to(style_mean, tf.shape(content_feature))

    style_std = expand_moments_dim(style_std)
    # TFLite does not support broadcasting; it is allowed for add, mul, sub, div
    # style_std = tf.broadcast_to(style_std, tf.shape(content_feature))

    normalized_content = tf.divide(content_feature - content_mean, content_std)
    tf.print(tf.reduce_sum(tf.cast(tf.math.is_nan(normalized_content), tf.int32)))
    return tf.multiply(normalized_content, style_std) + style_mean