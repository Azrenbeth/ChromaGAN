import tensorflow as tf
from keras import losses


def d_loss_fn(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss

d_loss_fn_3d = d_loss_fn

mse = losses.MeanSquaredError()
kld = losses.KLDivergence()


def g_loss_fn(img_ab_real, img_ab_fake, class_vector_real, class_vector_pred, gen_img_logits):
    mse_val = mse(img_ab_real, img_ab_fake)
    kld_val = kld(class_vector_real, class_vector_pred)
    was_val = tf.reduce_mean(gen_img_logits)
    return 1.0 * mse_val + 0.003 * kld_val - 0.1 * was_val

def merge_first_two_dims(tensor):
    target_shape = (-1,) + tuple(tensor.shape[2:])
    print(target_shape)
    return tf.reshape(tensor, shape=target_shape)

def g_loss_fn_3d(img_ab_real, img_ab_fake, class_vector_real, class_vector_pred, gen_img_logits):
    img_ab_real_ = merge_first_two_dims(img_ab_real)
    img_ab_fake_ = merge_first_two_dims(img_ab_fake)
    class_vector_real_ = merge_first_two_dims(class_vector_real)
    class_vector_pred_ = merge_first_two_dims(class_vector_pred)
    loss = g_loss_fn(img_ab_real_, img_ab_fake_, class_vector_real_, class_vector_pred_, gen_img_logits)

    # consistency loss
    img_ab_fake_1 = img_ab_fake[:, :-1]
    img_ab_fake_2 = img_ab_fake[:, 1:]
    img_ab_fake_1_ = merge_first_two_dims(img_ab_fake_1)
    img_ab_fake_2_ = merge_first_two_dims(img_ab_fake_2)
    loss += 0.2 * mse(img_ab_fake_1_, img_ab_fake_2_)
    return loss