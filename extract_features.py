import cv2
import numpy as np
import keras.backend as K
from keras.applications import ResNet50
from glob import glob
import os
from crop_champs import crop_champ_images
def preprocess_cv2(images):
    images = images.astype(np.float64)
    # # Zero-center by mean pixel
    images[:, :, :, 0] -= 103.939
    images[:, :, :, 1] -= 116.779
    images[:, :, :, 2] -= 123.68
    return images

def extract_resnet_features(x):
    net = ResNet50(include_top=False, weights='imagenet')
    #model = Model(input=net.input, output=net.get_layer(layer_name).output)
    result = net.predict(x)
    print("result of extract resnet feature", type(result), result.shape)
    return result

def l2_distance(A, B):
    return np.linalg.norm(A - B)


def extract_feature_images(images:list, h:int, w:int):
    """ Input: list of crop images, height and width size to extract Resnet feature
        Output: np.arr cotain feature of images"""
    h, w, c = h, w, 3
    # construct a batch
    batch = np.zeros(shape=(len(images), h, w, c))
    for i, image in enumerate(images):
        image_resize = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
        batch[i] = image_resize

    x = preprocess_cv2(batch)
    features = extract_resnet_features(x)
    return features


def extract_champions_features(images_fold, feature_fold, h, w):
    img_list = glob(os.path.join(images_fold, '*.jpg'))
    # resize both images such they they have the same size (so that the extracted features have the same dimension)
    h, w, c = h, w, 3
    # construct a batch
    batch = np.zeros(shape=(len(img_list), h, w, c))
    file_names = []
    for i, img_path in enumerate(img_list):
        file_name = img_path.split('/')[-1][:-4]
        file_names.append(file_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        batch[i] = img
    # preprocess the batch
    x = preprocess_cv2(batch)
    features = extract_resnet_features(x)
    print(features.shape)
    for i, file_name in enumerate(file_names):
        with open(os.path.join(feature_fold,'{}.nparray'.format(file_name)), 'wb') as f:
            np.save(f, features[i])


# def main():
#     images_fold = "/home/tlm/Documents/AI_Engineer_Test/test_data/test_images"
#     re = crop_champ_fold(images_fold)
#     rf = extract_feature_images(re, 320, 320, 16)

# if __name__ == "__main__":
#     main()