import cv2
import matplotlib.pyplot as plt
from PIL import  Image
import numpy as np
import random
import os
import cv2

def LoadImg(path, img_w=512, img_h=512):
    fullpath=path
    img = Image.open(fullpath)
    # img.resize((256, 256))
    img = img.resize((img_w, img_h))
    img=(np.array(img, dtype=np.float32)) / 255
    return img

def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        NoiseImg[randX, randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
    NoiseImg = np.clip(NoiseImg, 0, 1.0)
    return NoiseImg


# Check the output directory
def Checkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def gama_oneImgs(img, i):
    sigma_blur = random.uniform(0.3, 1.1)
    sigma_noise = random.uniform(0.1, 0.15)
    gamma =  random.uniform(1.8, 3.8)
    img = (np.array(img, dtype=np.float32)) / 255
    # print(img)
    print(i, ' -- sigma_blur=%s, sigma_noise=%s, gamma=%s '%(sigma_blur, sigma_noise, gamma))
    img = cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=sigma_blur)
    img = GaussianNoise(img, 0, sigma_noise, 0.02)
    img = np.power(img, gamma)
    return img


def gamma_oneList_op():
    input_path = 'E:/dim2rgb_imgs/gamaDataset256/gamaTest_rgbImgs256'
    out_path = 'E:/dim2rgb_imgs/gamaDataset256/gamaTest_dimImgs256'
    num = 0
    for img_name in os.listdir(input_path):
        num = num + 1
        img_path = input_path + '/' + img_name
        img = Image.open(img_path)
        # print(num, " -- img_path = ", img_path)
        outImg_name = img_name.split('rgb')[0] + 'dim'+img_name.split('rgb')[-1]
        gamma_path = out_path + '/' + outImg_name
        print(num, " -- gamma_path = ", gamma_path)
        gama_outImgs = gama_oneImgs(img, 0)
        plt.imsave(gamma_path, gama_outImgs)


if __name__ == "__main__":
    gamma_oneList_op()

