import tensorflow as tf
import tensorflow_datsets as tfds
from os.path import join

def get_image_from_coco(coco):
    image = coco['image']
    image = tf.cast(image, tf.float32)

    image_size = tf.shape(image)
    min_length = tf.reduce_min(image_size)
    image_size = image_size * 512 // min_length
    image = tf.image.resize(image, image_size)
    image = tf.image.random_crop(image, [256, 256, 3])
    return image

def get_coco_training_set():
    split = tfds.Split.TRAIN
    coco = tfds.load(name='coco/2017', split=split)
    return coco.map(get_image_from_coco)

def get_coco_test_set():
    split = tfds.Split.TEST
    coco = tfds.load(name='coco/2017', split=split)
    return coco.map(get_image_from_coco)

def get_image_from_wikiart(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image_size = tf.shape(image)
    min_length = tf.reduce_min(image_size)
    image_size = image_size * 512 // min_length
    image = tf.image.resize(image, image_size)
    image = tf.image.random_crop(image, [256, 256, 3])
    return image

def get_wikiart_set(file_dir):
    names = tf.data.Dataset.list_files(join(file_dir, "**/**/*.jpg"))
    images = names.map(get_image_from_wikiart)
    return images

def get_training_set(style_dir):
    coco_train = get_coco_training_set()
    wikiart_train = get_wikiart_set(style_dir)
    return tf.data.Dataset.zip((coco_train, wikiart_train))

def get_test_set(style_dir):
    coco_train = get_coco_test_set()
    wikiart_train = get_wikiart_set(style_dir)
    return tf.data.Dataset.zip((coco_train, wikiart_train))