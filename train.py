import argparse
import tensorflow as tf
from tqdm import tqdm
import os
from os.path import join, exists
from model import Net, get_decoder
from data import get_training_set, get_test_set
import pickle

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
parser.add_argument('-save_dir', type=str, default='saved_models')
parser.add_argument('-exp_name', type=str, required=True)
parser.add_argument('-lr', default=1e-4, type=float)
parser.add_argument('-lr_decay', default=5e-5, type=float)
parser.add_argument('-exp_name', type=str, required=True)
parser.add_argument('-batch_size', type=int , default=8)
parser.add_argument('-max_iter', type=int, default=160000)
parser.add_argument('-style_weight', type=float, default=10.0)
parser.add_argument('-content_weight', type=float, default=1.0)
parser.add_argument('-save_model_weights', type=str2bool, default=False)
parser.add_argument('-save_every', type=int , default=10000)
args = parser.parse_args()

encoder = tf.keras.applications.VGG19(include_top=False)
decoder = get_decoder()
model = Net(encoder, decoder)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    args.lr,
    decay_steps=1,
    decay_rate=args.lr_decay
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

train_data = get_training_set(args.style_dir).shuffle(30).batch(args.batch_size).repeat()
test_data = get_test_set(args.style_dir).batch(args.batch_size).repeat()

train_iter = iter(train_data)
test_iter = iter(test_data)

writer = tf.summary.SummaryWriter('logs/')

@tf.function
def train_step(content_images, style_images, step):
    with writer.as_default():
        with tf.GradientTape() as tape:
            loss_c, loss_s = model.train(content_images, style_images)
            loss = loss_c * args.content_weight + loss_s *  args.style_weight
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.summary.scalar("loss_c", loss_c, step=i)
        tf.summary.scalar("loss_s", loss_s, step=i)

with writer.as_default():
    for i in tqdm(range(args.max_iter)):
        content_images, style_images = next(train_iter)
        train_step(content_images, style_images, i)
        writer.flush()
        if (i+1) % args.save_every == 0 or (i+1) == args.max_iter:
            save_path = join(join(args.save_dir, args.exp_name), str(i))
            if not exists(save_path):
                os.makedirs(save_path)
            tf.saved_model.save(model, save_path)
            if args.save_model_weights:
                weight_path = join(join(args.save_dir, args.exp_name+'_weights'), str(i)+'_weights.h5')
                model.save_weights(weight_path, save_format='tf')

                # optimizer weights
                optimizer_file = join(join(args.save_dir, args.exp_name+'_weights'), str(i)+'_optimizer.pkl')
                optimizer_weights = getattr(optimizer, 'weights')
                weight_values = tf.keras.batch_get_value(optimizer_weights)
                with open(optimizer_file, 'wb') as f:
                    pickle.dump(weight_values, f)