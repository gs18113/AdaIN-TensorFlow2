import tensorflow as tf
from tensorflow import keras
from functions import adain

def get_decoder():
    return keras.Sequential([
        keras.layers.Conv2d(256, (3, 3), padding='same', activation='relu'),
        keras.layers.UpSampling2D(),
        keras.layers.Conv2d(256, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2d(256, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2d(256, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2d(128, (3, 3), padding='same', activation='relu'),
        keras.layers.UpSampling2D(),
        keras.layers.Conv2d(128, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2d(64, (3, 3), padding='same', activation='relu'),
        keras.layers.UpSampling2D(),
        keras.layers.Conv2d(64, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2d(3, (3, 3), padding='same', activation='relu'),
    ])

class Net(keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.enc_1 = keras.Sequential(encoder.layers[:2])
        self.enc_2 = keras.Seqeuntial(encoder.layers[2:5])
        self.enc_3 = keras.Sequential(encoder.layers[5:8])
        self.enc_4 = keras.Seqeuntial(encoder.layers[8:13])
        self.decoder = decoder
        self.mse = keras.losses.MeanSquaredError()

        # Freeze weights
        for enc_name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            getattr(self, enc_name).trainable = False

    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_%d' % (i+1))(input)
        return input

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_%d' % (i+1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
        return self.mse(input, target)

    # Calculate style loss of input & target(after going through VGG-19 layers)
    def calc_style_loss(self, input, target):
        input_mean, input_std = tf.nn.moments(input, axes=[1, 2])
        target_mean, target_std = tf.nn.moments(target, axes=[1, 2])
        return self.mse(input_mean, target_mean)+self.mse(input_std, target_std)

    @tf.function
    def call(self, content, style, alpha=1.0):
        style_feature = self.encode(style)
        content_feature = self.encode(content)
        t = adain(content_feature, style_feature)
        t = alpha * t + (1-alpha) * content_feature

        return self.decoder(t)
    
    def get_losses(self, content, style, alpha=1.0):
        style_features = self.encode_with_intermediate(style)
        content_feature = self.encode(content)
        t = adain(content_feature, style_features[-1])
        t = alpha * t + (1-alpha) * content_feature

        g_t = self.decoder(t)
        g_t_features = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_features[:-1], t)
        loss_s = 0
        for i in range(4):
            loss_s += self.calc_style_loss(g_t_features[i], style_features[i])
        return loss_c, loss_s