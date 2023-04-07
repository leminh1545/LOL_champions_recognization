import os
import cv2
import numpy as np
from glob import glob
import argparse

from crop_champs import crop_champ_images
from extract_features import extract_feature_images, l2_distance
from sklearn.preprocessing import normalize

dir_path = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Get path of input test folder")
parser.add_argument("--test_fold", type=str, help="Path to input test folder")
parser.add_argument("--batch_size", type=int, default = 16, help="Batch_size images to load, processing and predict")
args = parser.parse_args()

def predict(images:list, h:int, w:int, champions:dict):
    """ Input: raw images, height and width size to extract features, features extracted of champions label.
        Output: list of champ name prediction. """
    crop_faces = crop_champ_images(images)
    features = extract_feature_images(crop_faces, h, w)
    champs_data = list(champions.values())
    champs_key = list(champions.keys())
    result = [None] * len(images)
    for i, feature in enumerate(features):
        feature = feature.reshape(-1,1)
        feature_norm = normalize(feature, norm='l2', axis=0)
        dist_arr = np.zeros(len(champs_key))
        for j, champ_feature in enumerate(champs_data):
            champ_feature = champ_feature.reshape(-1,1)
            champ_norm = normalize(champ_feature, norm='l2', axis=0)
            distance = l2_distance(feature_norm, champ_norm)
            dist_arr[j] = distance
        predict_name = champs_key[np.argwhere(dist_arr == np.min(dist_arr)).item()]
        result[i] = predict_name
    return result

def predict_test(test_fold:list, h:int, w:int, batch_size:int, champions:dict):
    img_list = glob(os.path.join(test_fold, '*.jpg'))
    print("Founded {} images to test".format(len(img_list)))

    index = []
    result = {}
    result['file_name'] = []
    result['predict'] = []
    for i in range(0, len(img_list)):
    #     print(i)
        index.append(i)
        if(i % batch_size == batch_size - 1):
            print(index)
            # print("-", j)
            images = [None]*len(index)
            fns = [None]*len(index)
            for j in range(0, len(index)):
                img_path = img_list[index[j]]
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img_name = img_path.split('/')[-1][:-4]
                fns[j] = img_name
                images[j] = img
            batch_predict = predict(images, h, w, champions)
            result['file_name'].extend(fns)
            result['predict'].extend(batch_predict)
            index = []
        elif(i == len(img_list) -1):
            print(index)
            images = [None]*len(index)
            fns = [None]*len(index)
            for j in range(0, len(index)):
                img_path = img_list[index[j]]
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img_name = img_path.split('/')[-1][:-4]
                fns[j] = img_name
                images[j] = img
            batch_predict = predict(images, h, w, champions)
            result['file_name'].extend(fns)
            result['predict'].extend(batch_predict)
            # index = []
        else:
            continue
    return result

def main():
    ## Creat champion image features dict
    champ_features_fold = os.path.join(dir_path, "champions_features_embebded")
    champ_file_list = glob(os.path.join(champ_features_fold, '*'))
    champions = {}
    for file_path in champ_file_list:
        file_name = file_path.split('/')[-1][:-8]
        data = np.load(file_path)
        champions[file_name] =  data

    result_test = predict_test(args.test_fold, 320, 320, args.batch_size, champions)
    with open('test.txt', 'w') as f:
        for i, fn in enumerate(result_test['file_name']):
            f.write("%s\t%s\n" %(fn, result_test['predict'][i]))
if __name__ == "__main__":
    main()