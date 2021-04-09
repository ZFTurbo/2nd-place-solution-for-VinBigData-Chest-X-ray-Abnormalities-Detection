# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 4
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_common_functions import *


def convert_dicom_to_png(type='train', use_8bit=False, div=1):
    files = glob.glob(INPUT_PATH + '{}/*.dicom'.format(type))
    if use_8bit is False:
        if div != 1:
            out_folder = INPUT_PATH + '{}_png_16bit_div_{}/'.format(type, div)
        else:
            out_folder = INPUT_PATH + '{}_png_16bit/'.format(type)
    else:
        if div != 1:
            out_folder = INPUT_PATH + '{}_png_div_{}/'.format(type, div)
        else:
            out_folder = INPUT_PATH + '{}_png/'.format(type)

    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    for f in files:
        id = os.path.basename(f)[:-6]
        print('Go for: {}'.format(id))
        out_path = out_folder + '{}.png'.format(id)
        if os.path.isfile(out_path):
            continue
        img = read_xray(f, use_8bit=use_8bit, rescale_times=div)
        print(img.shape, img.min(), img.max(), img.dtype)
        cv2.imwrite(out_folder + '{}.png'.format(id), img)


if __name__ == '__main__':
    # convert_dicom_to_png('train', use_8bit=False, div=4)
    # convert_dicom_to_png('test', use_8bit=False, div=4)
    # convert_dicom_to_png('train', use_8bit=True, div=1)
    # convert_dicom_to_png('test', use_8bit=True, div=1)
    convert_dicom_to_png('train', use_8bit=True, div=2)
    convert_dicom_to_png('test', use_8bit=True, div=2)