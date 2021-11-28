import tensorflow as tf
from keras import losses

def d_loss_fn(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss

mse = losses.MeanSquaredError()
kld = losses.KLDivergence()

def g_loss_fn(img_ab_real, img_ab_fake, class_vector_real, class_vector_pred, gen_img_logits):
    mse_val = mse(img_ab_real, img_ab_fake)
    kld_val = kld(class_vector_real, class_vector_pred)
    was_val = tf.reduce_mean(gen_img_logits)
    return 1.0 * mse_val + 0.003 * kld_val - 0.1 * was_val
