import config
import os
import utils
import model
import losses

import dataClass as data
import dataClassDali as data_dali
import tensorflow as tf

from keras import applications
from tensorflow.keras.optimizers import Adam


# tf.compat.v1.disable_eager_execution()

def train(model, data, test_data):
    # Create folder to save models if needed.
    save_models_path = os.path.join(config.MODEL_DIR, config.TEST_NAME)
    if not os.path.exists(save_models_path):
        os.makedirs(save_models_path)

    # total number of batches in one epoch
    total_batch = int(data.size/config.BATCH_SIZE)

    for epoch in range(config.NUM_EPOCHS):
        for batch in range(total_batch):
            train_L, train_ab, train_L_3 = data.generate_batch()

            d_loss, g_loss = model.train_on_batch(
                [train_L, train_ab, train_L_3], [])

            print("[Epoch %d] [Batch %d/%d] [generator loss: %08f] [discriminator loss: %08f]" %
                  (epoch, batch, total_batch, g_loss, d_loss))

        # save models after each epoch
        # save_path = os.path.join(save_models_path, "my_model_colorization_epoch%d.h5" % epoch)
        # model.generator.save(save_path)

        # sample images after each epoch
        utils.sample_images(model.generator, test_data, epoch)


if __name__ == '__main__':
    # Create log folder
    log_path = os.path.join(config.LOG_DIR, config.TEST_NAME)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Load data
    print(f'Loading training data from {config.TRAIN_DIR}...')
    #train_data = data.DATA(config.TRAIN_DIR)
    train_data = data_dali.VideoDataLoader(
        config.TRAIN_DIR, config.IMAGE_SIZE, config.BATCH_SIZE, config.SEQUENCE_LENGTH, config.VIDEO_STRIDE)
    test_data = data.DATA(config.TEST_DIR)
    assert config.BATCH_SIZE <= train_data.size, "The batch size should be smaller or equal to the number of training images --> modify it in config.py"
    print("Training data loaded")

    # Create model
    print("Initializing model...")
    discriminator = model.get_discriminator()
    generator = model.get_generator()

    vgg_model = applications.vgg16.VGG16(weights='imagenet', include_top=True)
    wgan = model.WGAN(
        discriminator=discriminator,
        generator=generator,
        vgg_model=vgg_model
    )
    optimizer = Adam(0.00002, 0.5)
    wgan.compile(
        d_optimizer=optimizer,
        g_optimizer=optimizer,
        d_loss_fn=losses.d_loss_fn,
        g_loss_fn=losses.g_loss_fn
    )
    print("Model initialized")

    # Train
    print("Start training")
    train(wgan, train_data, test_data)
