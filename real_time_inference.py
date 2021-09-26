import cv2
import numpy as np
import ailia
import os
import shutil
import random

from util import log_init
from util.model_utils import check_and_download_models
from inference import human_seg

# ======================
# Parameters
# ======================
WEIGHT_PATH = 'u2net-human-seg.onnx'
MODEL_PATH = 'u2net-human-seg.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/u2net-human-seg/'
UNITY_PATH = f'{"/".join(os.getcwd().split("/")[:-3])}/unilab-tower-battle'
SAVE_IMAGE_PATH = f'{UNITY_PATH}/Assets/Resources'
IMAGE_SIZE = 320
RESULT_SIZE_RATIO = [0.25, 0.5, 0.75, 0.1]
RANDOM_RATIO = [0.3, 0.15, 0.05, 0.5]

logger = log_init.logger
logger.info('Start!')

# reset save image directory when run this code
shutil.rmtree(SAVE_IMAGE_PATH)
os.mkdir(SAVE_IMAGE_PATH)
print(UNITY_PATH)

def main():
    if not os.path.exists(f'{os.getcwd()}/{WEIGHT_PATH}') or not os.path.exists(f'{os.getcwd()}/{MODEL_PATH}'):
        check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # load model
    env_id = ailia.get_gpu_environment_id()
    logger.info(f'env_id: {env_id}')

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # capture camera
    cam = cv2.VideoCapture(1)
    count = 0
    while True:
        # get camera image
        _, img = cam.read()

        # show in the window
        cv2.imshow('PUSH SPACE KEY', img)

        # save image when input space key
        if cv2.waitKey(1) == 32:
            logger.info('Detect Space key and save camera image...')
            natural_image = img[:, :, ::-1]
            img = cv2.cvtColor(natural_image, cv2.COLOR_BGRA2RGB)

            # inference
            logger.info('Start inference...')
            pred = human_seg(net, img) * 255

            # segment original image
            predict = (pred > 128) * 255
            predict = cv2.bitwise_not(predict)
            ret, markers = cv2.connectedComponents(np.uint8(predict), connectivity=4)
            r_channel, g_channel, b_channel = cv2.split(natural_image)
            alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype)
            alpha_channel[markers != 1] = [255]
            result = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
            ratio = random.choices(RESULT_SIZE_RATIO, k=1, weights=RANDOM_RATIO)
            result = cv2.resize(result, dsize=None, fx=ratio[0], fy=ratio[0], interpolation=cv2.INTER_AREA)

            cv2.imwrite(f'{SAVE_IMAGE_PATH}/image_{count}.png', result)
            logger.info('Script finished successfully.')
            count += 1

        # break when input enter key
        if cv2.waitKey(1) == 13:
            logger.info('Detect Enter key and finish this project...')
            break

    # stop camera
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
