import config
import tensorflow as tf

from tensorflow import keras
from keras.layers import Input
from keras.models import Model
from keras import applications
from keras.layers import advanced_activations

class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        vgg_model,
        gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.vgg_model = vgg_model
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_ab, fake_ab, img_L):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_ab - real_ab
        interpolated = real_ab + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([interpolated, img_L], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        inputs, _ = data
        img_L, img_ab_real, img_L_3 = inputs
        class_vector_real = self.vgg_model(img_L_3)

        # Get the batch size
        batch_size = tf.shape(img_L)[0]

        # Train the generator
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            img_ab_fake, class_vector_pred = self.generator(img_L_3, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator([img_ab_fake, img_L], training=False)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(img_ab_real, img_ab_fake, class_vector_real, class_vector_pred, gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        # Train the discriminator
        with tf.GradientTape() as tape:
            # Generate fake images
            img_ab_fake, _ = self.generator(img_L_3, training=False)
            # Get the logits for the fake images
            fake_logits = self.discriminator([img_ab_fake, img_L], training=True)
            # Get the logits for the real images
            real_logits = self.discriminator([img_ab_real, img_L], training=True)

            # Calculate the discriminator loss using the fake and real image logits
            d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
            # Calculate the gradient penalty
            gp = self.gradient_penalty(batch_size, img_ab_real, img_ab_fake, img_L)
            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        return {"d_loss": d_loss, "g_loss": g_loss}

img_shape_1 = (config.IMAGE_SIZE, config.IMAGE_SIZE, 1)
img_shape_2 = (config.IMAGE_SIZE, config.IMAGE_SIZE, 2)
img_shape_3 = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)

def get_discriminator():
    input_ab = Input(shape=img_shape_2, name='ab_input')
    input_l = Input(shape=img_shape_1, name='l_input')
    net = keras.layers.concatenate([input_l, input_ab])
    net = keras.layers.Conv2D(
        64, (4, 4), padding='same', strides=(2, 2))(net)  # 112, 112, 64
    net = advanced_activations.LeakyReLU()(net)
    net = keras.layers.Conv2D(
        128, (4, 4), padding='same', strides=(2, 2))(net)  # 56, 56, 128
    net = advanced_activations.LeakyReLU()(net)
    net = keras.layers.Conv2D(
        256, (4, 4), padding='same', strides=(2, 2))(net)  # 28, 28, 256
    net = advanced_activations.LeakyReLU()(net)
    net = keras.layers.Conv2D(
        512, (4, 4), padding='same', strides=(1, 1))(net)  # 28, 28, 512
    net = advanced_activations.LeakyReLU()(net)
    net = keras.layers.Conv2D(
        1, (4, 4), padding='same', strides=(1, 1))(net)  # 28, 28,1
    return Model([input_ab, input_l], net)

def get_generator():
    input_img = Input(shape=img_shape_3)

    # VGG16 without top layers
    VGG_model = applications.vgg16.VGG16(
        weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model_ = Model(VGG_model.input, VGG_model.layers[-6].output)
    model = model_(input_img)

    # Global Features

    global_features = keras.layers.Conv2D(
        512, (3, 3), padding='same', strides=(2, 2), activation='relu')(model)
    global_features = keras.layers.BatchNormalization()(global_features)
    global_features = keras.layers.Conv2D(
        512, (3, 3), padding='same', strides=(1, 1), activation='relu')(global_features)
    global_features = keras.layers.BatchNormalization()(global_features)

    global_features = keras.layers.Conv2D(
        512, (3, 3), padding='same', strides=(2, 2), activation='relu')(global_features)
    global_features = keras.layers.BatchNormalization()(global_features)
    global_features = keras.layers.Conv2D(
        512, (3, 3), padding='same', strides=(1, 1), activation='relu')(global_features)
    global_features = keras.layers.BatchNormalization()(global_features)

    global_features2 = keras.layers.Flatten()(global_features)
    global_features2 = keras.layers.Dense(1024)(global_features2)
    global_features2 = keras.layers.Dense(512)(global_features2)
    global_features2 = keras.layers.Dense(256)(global_features2)
    global_features2 = keras.layers.RepeatVector(28*28)(global_features2)
    global_features2 = keras.layers.Reshape(
        (28, 28, 256))(global_features2)

    global_featuresClass = keras.layers.Flatten()(global_features)
    global_featuresClass = keras.layers.Dense(4096)(global_featuresClass)
    global_featuresClass = keras.layers.Dense(4096)(global_featuresClass)
    global_featuresClass = keras.layers.Dense(
        1000, activation='softmax')(global_featuresClass)

    # Midlevel Features

    midlevel_features = keras.layers.Conv2D(
        512, (3, 3),  padding='same', strides=(1, 1), activation='relu')(model)
    midlevel_features = keras.layers.BatchNormalization()(midlevel_features)
    midlevel_features = keras.layers.Conv2D(256, (3, 3),  padding='same', strides=(
        1, 1), activation='relu')(midlevel_features)
    midlevel_features = keras.layers.BatchNormalization()(midlevel_features)

    # fusion of (VGG16 + Midlevel) + (VGG16 + Global)
    modelFusion = keras.layers.concatenate(
        [midlevel_features, global_features2])

    # Fusion + Colorization
    outputModel = keras.layers.Conv2D(
        256, (1, 1), padding='same', strides=(1, 1), activation='relu')(modelFusion)
    outputModel = keras.layers.Conv2D(
        128, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)

    outputModel = keras.layers.UpSampling2D(size=(2, 2))(outputModel)
    outputModel = keras.layers.Conv2D(
        64, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)
    outputModel = keras.layers.Conv2D(
        64, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)

    outputModel = keras.layers.UpSampling2D(size=(2, 2))(outputModel)
    outputModel = keras.layers.Conv2D(
        32, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)
    outputModel = keras.layers.Conv2D(2, (3, 3), padding='same', strides=(
        1, 1), activation='sigmoid')(outputModel)
    outputModel = keras.layers.UpSampling2D(size=(2, 2))(outputModel)
    final_model = Model(inputs=[input_img], outputs=[
                        outputModel, global_featuresClass])

    return final_model
