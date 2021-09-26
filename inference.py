import sys
import cv2
import numpy as np
import ailia
import os

from util import log_init
from skimage import transform
from util.model_utils import check_and_download_models  # noqa: E402


logger = log_init.logger
logger.info('Start!')
# ======================
# Parameters
# ======================
args = sys.argv

WEIGHT_PATH = 'u2net-human-seg.onnx'
MODEL_PATH = 'u2net-human-seg.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/u2net-human-seg/'

IMAGE_PATH = f'images/source_images/{args[1]}'
SAVE_IMAGE_PATH = 'images/result_images/'
IMAGE_SIZE = 320

if not os.path.exists(os.path.join(os.getcwd(), SAVE_IMAGE_PATH)):
    os.makedirs(SAVE_IMAGE_PATH)
SAVE_IMAGE_PATH += f'{IMAGE_PATH.split("/")[-1]}'
# ======================
# Utils
# ======================
def preprocess(img):
    img = transform.resize(img, (IMAGE_SIZE, IMAGE_SIZE), mode='constant')

    img = img / np.max(img)
    img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
    img = img.astype(np.float32)

    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


def preprocessing_img(img):
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    return img


def load_image(image_path):
    if os.path.isfile(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        logger.error(f'{image_path} not found.')
        sys.exit()
    return preprocessing_img(img)

# ======================
# Main functions
# ======================
def human_seg(net, img):
    h, w = img.shape[:2]

    # initial preprocesses
    img = preprocess(img)

    # feedforward
    output = net.predict([img])
    d1, d2, d3, d4, d5, d6, d7 = output
    pred = d1[:, 0, :, :]

    # post processes
    ma = np.max(pred)
    mi = np.min(pred)
    pred = (pred - mi) / (ma - mi)

    pred = pred.transpose(1, 2, 0)  # CHW -> HWC
    pred = cv2.resize(pred, (w, h), cv2.INTER_LINEAR)

    return pred


def recognize_from_image(net):
    # prepare input data
    logger.info(IMAGE_PATH)
    natural_image = cv2.imread(IMAGE_PATH)[:, :, ::-1]
    img = load_image(IMAGE_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # inference
    logger.info('Start inference...')
    pred = human_seg(net, img)
    res_img = pred * 255

    # segment original image
    predict = (res_img > 128) * 255
    predict = cv2.bitwise_not(predict)
    ret, markers = cv2.connectedComponents(np.uint8(predict), connectivity=4)
    r_channel, g_channel, b_channel = cv2.split(natural_image)
    alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype)
    alpha_channel[markers != 1] = [255]
    result = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    cv2.imwrite(SAVE_IMAGE_PATH, result)

    logger.info('Script finished successfully.')


def main():
    if not os.path.exists(f'{os.getcwd()}/{WEIGHT_PATH}') or not os.path.exists(f'{os.getcwd()}/{MODEL_PATH}'):
        check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # load model
    env_id = ailia.get_gpu_environment_id()
    logger.info(f'env_id: {env_id}')

    # initialize
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    # image mode
    recognize_from_image(net)


if __name__ == '__main__':
    main()
