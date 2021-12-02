import os
import config
import dataClassDali as data
import tensorflow as tf
import keras.models
import utils
import model
import cv2
import numpy as np
from tqdm import tqdm

SEQUENCE_LENGTH=4
BATCH_SIZE=4
STRIDE=1

if __name__=='__main__':
    model_filename = os.path.join(config.MODEL_DIR, config.TEST_NAME, config.PRETRAINED)
    model = keras.models.load_model(model_filename, custom_objects={ "GeneratorCell": model.GeneratorCell })

    test_data_dir = os.path.join(config.DATA_DIR, config.TEST_DIR)
    test_data = data.VideoDataLoader(test_data_dir, config.IMAGE_SIZE, BATCH_SIZE, SEQUENCE_LENGTH, STRIDE, combine_batch_and_seq=not config.TEMPORAL_CONSISTENCY, random_shuffle=False)

    out_filename = os.path.join(config.OUT_DIR, config.TEST_NAME, "output.mp4")
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    print("Output filename: " + out_filename)
    vid_writer = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc("P", "I", "M", "1"), 30, (config.IMAGE_SIZE, config.IMAGE_SIZE))

    for j in tqdm(range(test_data.size // BATCH_SIZE)):
        # TODO: pass the state from previous batch to the next batch
        train_L, _, train_L_3 = test_data.generate_batch()
        train_L = tf.reshape(train_L, (-1,) + tuple(train_L.shape[-3:])).numpy()
        pred_ab, _ = model.predict(train_L_3)
        pred_ab = tf.reshape(pred_ab, (-1,) + tuple(pred_ab.shape[-3:])).numpy()
        
        for i in range(pred_ab.shape[0]):
            pred_ab_frame = utils.deprocess(pred_ab[i])
            train_L_frame = utils.deprocess(train_L[i])
            result = np.concatenate((train_L_frame, pred_ab_frame), axis=2)
            result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
            vid_writer.write(result)
    
    vid_writer.release()
