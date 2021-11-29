import config
import numpy as np
import cv2
import os


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def reconstruct(batchX, predictedY, filelist):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    save_results_path = os.path.join(config.OUT_DIR, config.TEST_NAME)
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
    save_path = os.path.join(
        save_results_path, filelist + "_reconstructed.jpg")
    cv2.imwrite(save_path, result)
    return result


def reconstruct_no(batchX, predictedY):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    return result


def sample_images(colorization_model, test_data, epoch, three_dim=False):
    # reset index so get same images each time
    test_data.data_index = 0

    # load test data
    testL, _,  filelist, original, labimg_oritList = test_data.generate_batch()

    # predict AB channels
    predAB, _ = colorization_model.predict(
        np.tile(testL, [1, 1, 1, 3]))

    # print results
    for i in range(test_data.batch_size):
        originalResult = original[i]
        height, width, channels = originalResult.shape
        predictedAB = cv2.resize(deprocess(predAB[i]), (width, height))
        labimg_ori = np.expand_dims(labimg_oritList[i], axis=2)
        predResult = reconstruct(
            deprocess(labimg_ori), predictedAB, "epoch"+str(epoch)+"_"+filelist[i][:-4])
