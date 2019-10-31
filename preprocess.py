import tensorflow as tf
import os, glob
from os.path import join
from multiprocessing import Pool
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-style_dir', type=str, default='style_images')
parser.add_argument('-output_file', type=str, default='processed_images/processed.tfrecords')
args = parser.parse_args()

def asdf(filename):
    try:
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image_size = tf.shape(image)[:2]
        min_length = tf.reduce_min(image_size)
        image_size = image_size * 512 // min_length
        image = tf.image.resize(image, image_size)
        image = tf.image.random_crop(image, [256, 256, 3])
        return tf.io.serialize_tensor(image)
    except:
        return None

record_file = join(args.output_file, 'processed.tfrecords')
style_path = join(args.style_dir, '**/**/*.jpg')
count = 0
with tf.io.TFRecordWriter(record_file) as writer:
    with Pool(8) as pool:
        for image in pool.imap(tqdm(glob.glob(style_path))):
            if image != None:
                writer.write(image)
                count += 1

print('Total iamge count: %d' % count)