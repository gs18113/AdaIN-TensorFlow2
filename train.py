import argparse
import tensorflow as tf
from tqdm import tqdm
import os, glob
from os.path import join, exists
from model import Net, get_decoder
from data import get_training_set, get_test_set
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-style_dir', type=str, default='style_images')
parser.add_argument('-output_dir', type=str, default='outputs')
parser.add_argument('-lr', default=1e-5, type=float)
parser.add_argument('-lr_decay', default=5e-5, type=float)
parser.add_argument('-exp_name', type=str, required=True)
parser.add_argument('-batch_size', type=int , default=8)
parser.add_argument('-max_iter', type=int, default=160000)
parser.add_argument('-style_weight', type=float, default=10.0)
parser.add_argument('-content_weight', type=float, default=1.0)
parser.add_argument('-save_tflite', type=str2bool, default=False)
parser.add_argument('-save_every', type=int , default=10000)
parser.add_argument('-image_every', type=int , default=500)
args = parser.parse_args()

tf.random.set_seed(123)

encoder = tf.keras.applications.VGG19(include_top=False)
decoder = get_decoder()
model = Net(encoder, decoder)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    args.lr,
    decay_steps=1,
    decay_rate=args.lr_decay
)
#optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=0.5)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

train_data = get_training_set(args.style_dir).repeat().shuffle(30).batch(args.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

train_iter = iter(train_data)

writer_path = join(args.output_dir, args.exp_name, 'logs/')
if not exists(writer_path):
    os.makedirs(writer_path)
writer = tf.summary.create_file_writer(writer_path)

@tf.function
def train_step(content_images, style_images):
    with tf.GradientTape() as tape:
        loss_c, loss_s = model.train_batch(content_images, style_images)
        loss = loss_c * args.content_weight + loss_s *  args.style_weight
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_c, loss_s

# Checkpoints
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)

ckpt_dir = join(args.output_dir, args.exp_name, 'ckpts/')
if not exists(ckpt_dir):
    os.makedirs(ckpt_dir)
manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=100)

# Need to run on sample input to generate graph
if args.save_tflite:
    sample_input_0 = tf.random.uniform([1, 256, 256, 3], dtype=tf.float32)
    sample_input_1 = tf.random.uniform([1, 256, 256, 3], dtype=tf.float32)
    sample_input_2 = tf.random.uniform([], dtype=tf.float32)
    sample_output = model(sample_input_0, sample_input_1, sample_input_2)

logging.info('All ready, starting train steps')
with writer.as_default():
    for i in tqdm(range(args.max_iter)):
        content_images, style_images = next(train_iter)
        loss_c, loss_s = train_step(content_images, style_images)
        tf.summary.scalar("loss_c", loss_c, step=i)
        tf.summary.scalar("loss_s", loss_s, step=i)
        writer.flush()
        ckpt.step.assign_add(1)
        if (i+1) % args.image_every == 0:
            logging.info('Writing example images...')
            content_images, style_images = next(train_iter)
            tf.summary.image("content", content_images, step=i)
            tf.summary.image("style", style_images, step=i)
            tf.summary.image("output", model(content_images, style_images, tf.ones(())), step=i)
            logging.info('Writing complete')
        if (i+1) % args.save_every == 0 or (i+1) == args.max_iter:
            logging.info('Saving model checkpoint...')
            ckpt_path = manager.save()
            logging.info('Saved model checkpoint to %s' % ckpt_path)
            #weight_file = join(args.output_dir, args.exp_name+'_weights', str(i)+'_weights.h5')
            # logging.info('Saving weights to %s' % weight_file)

            # model.save_weights(weight_file, save_format='tf')

            # # optimizer weights
            # optimizer_file = join(args.output_dir, args.exp_name+'_weights', str(i)+'_optimizer.pkl')
            # logging.info('Saving optimizer state to %s' % weight_file)

            # weight_values = optimizer.get_weights()
            # with open(optimizer_file, 'wb') as f:
            #     pickle.dump(weight_values, f)
            if args.save_tflite:
                tflite_path = join(args.output_dir, args.exp_name, 'tflite')
                if not exists(tflite_path):
                    os.makedirs(tflite_path)
                tflite_file = join(tflite_path, args.exp_name+'_'+str(i)+'.tflite')
                logging.info('Saving TFLite model...')
                tf_fn = tf.function(model.call, input_signature=(tf.TensorSpec(shape=(
                    None, 256, 256, 3)), tf.TensorSpec(shape=(None, 256, 256, 3)), tf.TensorSpec(shape=())))
                converter = tf.lite.TFLiteConverter.from_concrete_functions([tf_fn.get_concrete_function()])
                tflite_model = converter.convert()
                open(tflite_file, 'wb').write(tflite_model)
                logging.info('Saved TFLite model to %s' % tflite_file)
