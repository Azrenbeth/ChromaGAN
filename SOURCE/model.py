from tensorflow.python.framework.tensor_shape import TensorShape
import config
import tensorflow as tf
import keras.layers

from tensorflow import keras
from keras.layers import Input
from keras.models import Model
from keras import applications
from keras.layers import advanced_activations

import transformerBlocks as trans


class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        vgg_model,
        gp_weight=10.0,
        three_dim=False
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.vgg_model = vgg_model
        self.gp_weight = gp_weight
        self.three_dim = three_dim

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_ab, fake_ab, img_L, sequence_length=0):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        if not self.three_dim:
            alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        else:
            alpha = tf.random.uniform([batch_size, sequence_length, 1, 1, 1], 0.0, 1.0)
        
        diff = fake_ab - real_ab
        interpolated = real_ab + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([interpolated, img_L], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3] if not self.three_dim else [2, 3, 4]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        inputs, _ = data
        img_L, img_ab_real, img_L_3 = inputs
        class_vector_real = self.vgg_model(img_L_3)

        # Get the batch size and the sequence length
        batch_size = tf.shape(img_L)[0]

        sequence_length = None
        if self.three_dim:
            sequence_length = tf.shape(img_L)[1]

        # Train the generator
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            img_ab_fake, class_vector_pred = self.generator(
                img_L_3, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(
                [img_ab_fake, img_L], training=False)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(
                img_ab_real, img_ab_fake, class_vector_real, class_vector_pred, gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(
            g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        # Train the discriminator
        with tf.GradientTape() as tape:
            # Generate fake images
            img_ab_fake, _ = self.generator(img_L_3, training=False)
            # Get the logits for the fake images
            fake_logits = self.discriminator(
                [img_ab_fake, img_L], training=True)
            # Get the logits for the real images
            real_logits = self.discriminator(
                [img_ab_real, img_L], training=True)

            # Calculate the discriminator loss using the fake and real image logits
            d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
            # Calculate the gradient penalty
            gp = self.gradient_penalty(
                batch_size, img_ab_real, img_ab_fake, img_L, sequence_length)
            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(
                d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        return {"d_loss": d_loss, "g_loss": g_loss}


img_shape_1 = (config.IMAGE_SIZE, config.IMAGE_SIZE, 1)
img_shape_2 = (config.IMAGE_SIZE, config.IMAGE_SIZE, 2)
img_shape_3 = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)


def get_conv_discriminator():
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


def get_trans_discriminator():
    input_ab = Input(shape=img_shape_2, name='ab_input')
    input_l = Input(shape=img_shape_1, name='l_input')
    net = keras.layers.concatenate([input_l, input_ab])

    patch_size = 28
    num_patches_1d = int(config.IMAGE_SIZE / patch_size)
    num_patches = num_patches_1d ** 2
    embedding_dimensions = patch_size ** 2
    num_heads = num_patches

    patches = trans.Patches(patch_size)(net)
    encoded_patches = trans.PatchEncoder(
        num_patches, embedding_dimensions)(patches)

    for _ in range(4):
        encoded_patches = trans.TransformerBlock(
            num_heads,
            embedding_dimensions,
            dropout_rate=0.1,
        )(encoded_patches)

    representation = keras.layers.LayerNormalization(
        epsilon=1e-6)(encoded_patches)
    representation = keras.layers.Dropout(0.5)(representation)

    features = trans.mlp(
        representation,
        hidden_units=[512, 64],
        dropout_rate=0.5
    )

    classification = tf.reshape(
        keras.layers.Dense(1)(features),
        (-1, num_patches_1d, num_patches_1d, 1)
    )

    return Model(inputs=[input_ab, input_l], outputs=classification)


def get_generator(with_interframe_conn=False):
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
    
    if with_interframe_conn:
        last_model_fusion = Input(shape=(28, 28, 512), name='last_model_fusion')
        modelFusion = keras.layers.concatenate([modelFusion, last_model_fusion])
        modelFusion = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='tanh')(modelFusion)

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

    if with_interframe_conn:
        final_model = Model(inputs=[input_img, last_model_fusion], outputs=[
                            outputModel, global_featuresClass, modelFusion], name="generator_w_interframe_conn")
    else:
        final_model = Model(inputs=[input_img], outputs=[
                            outputModel, global_featuresClass])

    return final_model

class GeneratorCell(keras.layers.Layer):
    def __init__(self, model, **kwargs):
        self.model = model
        self.state_size = [tf.TensorShape((28, 28, 512))],
        self.output_size = [tf.TensorShape(img_shape_2), tf.TensorShape((1000,))]
        super(GeneratorCell, self).__init__(**kwargs)
    
    def build(self, input_shapes):
        self.model.build(input_shapes)

    def get_config(self):
        config = super().get_config()
        config.update({
            "model": self.model
        })
        return config

    def call(self, inputs, states):
        output_image, output_class, new_state = self.model([inputs, states[0]])
        return (output_image, output_class), (new_state,)

def get_3d_generator():
    input_img = Input(shape=(None,) + img_shape_3)
    outputs = keras.layers.RNN(GeneratorCell(get_generator(True)), return_sequences=True)(input_img)
    model = Model(inputs=[input_img], outputs=outputs)
    return model

def get_3d_discriminator(raw_discriminator):
    # HACK: use RNN to work with the additional timestep dimension. Any better ways?
    class DiscriminatorCell(keras.layers.Layer):
        def __init__(self, model, **kwargs):
            self.model = model
            self.state_size = []
            self.output_size = [TensorShape((28, 28, 1))]
            super(DiscriminatorCell, self).__init__(**kwargs)
        
        def build(self, input_shapes):
            self.model.build(input_shapes)
        
        def get_config(self):
            config = super().get_config()
            config.update({
                "model": self.model
            })
            return config
        
        def call(self, inputs, states):
            output = self.model(inputs)
            return output, ()

    input_l = Input(shape=(None,) + img_shape_1)
    input_ab = Input(shape=(None,) + img_shape_2)
    classification = keras.layers.RNN(DiscriminatorCell(raw_discriminator), return_sequences=True)(inputs=(input_ab, input_l))
    model = Model(inputs=[input_ab, input_l], outputs=[classification])
    return model

def get_vgg():
    return applications.vgg16.VGG16(weights='imagenet', include_top=True)


# HACK: use RNN to work with the additional timestep dimension. Any better ways?
class VGGCell(keras.layers.Layer):
    def __init__(self, model, **kwargs):
        self.model = model
        self.state_size = []
        self.output_size = [TensorShape((1000,))]
        super(VGGCell, self).__init__(**kwargs)
    
    def build(self, input_shapes):
        self.model.build(input_shapes)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "model": self.model
        })
        return config
    
    def call(self, inputs, states):
        output = self.model(inputs)
        return output, ()
        
def get_3d_vgg():
    raw_vgg = get_vgg()
    input_l3 = Input(shape=(None,) + img_shape_3)
    classification = keras.layers.RNN(VGGCell(raw_vgg), return_sequences=True)(inputs=input_l3)
    model = Model(inputs=[input_l3], outputs=[classification])
    return model