import tensorflow as tf

def expand_moments_dim(moment):
    return tf.reshape(moment, [-1, 1, 1, 3])

def adain(content_feature, style_feature):
    content_mean, content_std = tf.nn.moments(content_feature, axes=[1, 2])
    style_mean, style_std = tf.nn.moments(style_feature, axes=[1, 2])

    content_mean = expand_moments_dim(content_mean)
    content_mean = tf.broadcast_to(content_mean, tf.shape(content_feature))

    content_std = expand_moments_dim(content_std)
    content_std = tf.broadcast_to(content_std, tf.shape(content_feature))

    style_mean = expand_moments_dim(style_mean)
    style_mean = tf.broadcast_to(style_mean, tf.shape(content_feature))

    style_std = expand_moments_dim(style_std)
    style_std = tf.broadcast_to(style_std, tf.shape(content_feature))

    normalized_content = (content_feature - content_mean) / content_std
    return normalized_content * style_std + style_mean