# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 4
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_common_functions import *
import pydicom


def extract_meta(type='train'):
    files = glob.glob(INPUT_PATH + '{}/*.dicom'.format(type))
    dt = []
    for f in files:
        print('Go for: {}'.format(f))
        id = os.path.basename(f)[:-6]
        dicom = pydicom.dcmread(f)
        dict1 = get_dicom_as_dict(dicom)
        dict1['image_id'] = id
        dt.append(dict1)
    s = pd.DataFrame(dt)
    s.to_csv(OUTPUT_PATH + 'dicom_properties_{}.csv'.format(type), index=False)

    print(s.describe())


def extract_width_height(type='train'):
    files = glob.glob(INPUT_PATH + '{}/*.dicom'.format(type))
    out = open(OUTPUT_PATH + 'image_width_height_{}.csv'.format(type), 'w')
    out.write('image_id,width,height\n')
    for f in files:
        print('Go for: {}'.format(f))
        dicom = pydicom.dcmread(f)
        image = dicom.pixel_array
        height, width = image.shape
        out.write('{},{},{}\n'.format(os.path.basename(f)[:-6], width, height))
    out.close()


def extract_width_height_external(type='external'):
    files = glob.glob(INPUT_PATH + 'nih/*.png'.format(type))
    out = open(OUTPUT_PATH + 'image_width_height_{}.csv'.format(type), 'w')
    out.write('image_id,width,height\n')
    for f in files:
        print('Go for: {}'.format(f))
        image = cv2.imread(f, 0)
        height, width = image.shape
        out.write('{},{},{}\n'.format(os.path.basename(f)[:-4], width, height))
    out.close()


if __name__ == '__main__':
    extract_meta('train')
    extract_meta('test')
    extract_width_height('train')
    extract_width_height('test')
    # extract_width_height_external()