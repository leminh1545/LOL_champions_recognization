import os
from glob import glob
import cv2
import numpy as np


def sliding_window(image, stepSize, windowSize):
    windows = []
    for x in range(0, image.shape[1] - windowSize[0] + 1, stepSize):
        windows.append(image[0:windowSize[1], x:x + windowSize[0]])
    return windows

def crop_champ_face(img):
    """ Input: one raw image
        Output: list contain one face image
        In case that test images different with present images, 
        can use sliding windows to crop retangel champ faces."""
    h,w,c = img.shape
    # print("shape:", h,w,c)
    image_crop = img[0:h, 0:int(w/2)]
    
    lab= cv2.cvtColor(image_crop, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray,10,50,50)       
    
    detected_circles = cv2.HoughCircles(bilateral, 
                       cv2.HOUGH_GRADIENT_ALT, 1.25, 10, param1 = 300,
                   param2 = 0.75, minRadius = 15, maxRadius = 100)
    crop_faces = []
    # Draw circles that are detected.
    if detected_circles is not None:
        try:
            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))
            pt = detected_circles[0][0]
            a, b, r = pt[0], pt[1], pt[2]
            crop_face = img[b-r:b+r, a-r:a+r]
            crop_faces.append(crop_face)
            return crop_faces
        except Exception as e:
            print(e)
            return None
    else:
        image_resize = cv2.resize(image_crop, (100,60)) #w100, h60
        windows = sliding_window(image_resize, 20, (60, 60)) #w60, h60
        crop_faces.append(windows[1]) ## In case raw test images similar with test image present
        return crop_faces
def crop_champ_images(images:list):
    """ Input: Raw images
        Output: List contain champ face images (np.arr)"""
    crop_champ_faces = []
    for i, image in enumerate(images):
        crop_faces = crop_champ_face(image)
        crop_champ_faces.append(crop_faces[0])
    return crop_champ_faces
# def crop_champ_fold(fold_path):
#     image_list = glob(os.path.join(fold_path, '*.jpg'))
#     print("Num images test fold", len(image_list))
#     croped_faces = {}
#     for i, img_path in enumerate(image_list):
#         img_name = img_path.split('/')[-1][:-4]
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         croped_face = crop_champ_face(img)
#         # print(croped_face[0].shape)
#         croped_faces[str(img_name)] = croped_face[0]
#         # cv2.imshow("images", croped_face[0])
#         # k = cv2.waitKey(0)
#         # if k== ord('b'):
#         #     cv2.destroyWindow("images")
#         #     break
#         # elif k== ord('q'): 
#         #     cv2.destroyWindow("images")
#         #     continue
#     return croped_faces
    
# def main():
    # images_fold = "/home/tlm/Documents/AI_Engineer_Test/test_data/test_images"
    # re = crop_champ_fold(images_fold)
# if __name__ == "__main__":
#     main()