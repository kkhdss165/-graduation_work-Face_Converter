import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torchvision import transforms, models
import cv2
import os
from glob import glob
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_CLASSES = 20
NUM_TRAIN_IMAGES = 1000
NUM_VAL_IMAGES = 50


colormap = loadmat("./convert/source/models/human_colormap.mat")["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)

labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
class removeBack:
    def __init__(self):
        self.model_1 = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
        self.cmap = plt.cm.get_cmap('tab20c')
        self.colors = (self.cmap(np.arange(self.cmap.N)) * 255).astype(np.int)[:, :3].tolist()
        np.random.seed(2020)
        np.random.shuffle(self.colors)
        self.colors.insert(0, [0, 0, 0]) # background color must be black
        self.colors = np.array(self.colors, dtype=np.uint8)

        self.palette_map = np.empty((10, 0, 3), dtype=np.uint8)
        self.legend = []

        for i in range(21):
            self.legend.append(mpatches.Patch(color=np.array(self.colors[i]) / 255., label='%d: %s' % (i, labels[i])))
            self.c = np.full((10, 10, 3), self.colors[i], dtype=np.uint8)
            self.palette_map = np.concatenate([self.palette_map, self.c], axis=1)

    def segment(self, net, img):
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        img = img[:,:,:3]
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model_1.to('cuda')

        output = self.model_1(input_batch)['out'][0] # (21, height, width)

        output_predictions = output.argmax(0).byte().cpu().numpy() # (height, width)

        r = Image.fromarray(output_predictions).resize((img.shape[1], img.shape[0]))
        r.putpalette(self.colors)

        return r, output_predictions

    def changeBGR(self, image, size):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size))
        return image

    def getremovemask(self,file_name):

        img = np.array(Image.open(file_name))
        fg_h, fg_w, _ = img.shape
        segment_map, pred = self.segment(self.model_1, img)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        numpy_image=np.array(segment_map)
        opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        _, mask = cv2.threshold(opencv_image,10,255, cv2.THRESH_BINARY)

        img = self.changeBGR(img, (fg_w, fg_h))
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        return mask
class hairsegement:
    def __init__(self):
        self.model2 = tf.keras.models.load_model('./convert/source/models/model3.h5')

    def read_image2(self,image, mask=False):
        image = np.asarray(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.convert_to_tensor(rgb, dtype=tf.float32)

        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 127.5 - 1

        return image

    def infer(self,model, image_tensor):
        predictions = model.predict(np.expand_dims((image_tensor), axis=0))
        predictions = np.squeeze(predictions)
        predictions = np.argmax(predictions, axis=2)
        return predictions

    def decode_segmentation_masks(self,mask, colormap, n_classes):
        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        for l in range(0, n_classes):
            idx = mask == l
            r[idx] = colormap[l, 0]
            g[idx] = colormap[l, 1]
            b[idx] = colormap[l, 2]
        rgb = np.stack([r, g, b], axis=2)
        return rgb

    def get_overlay(self,image, colored_mask):
        image = tf.keras.preprocessing.image.array_to_img(image)
        image = np.array(image).astype(np.uint8)
        overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
        return overlay

    def plot_predictions2(self,img, colormap, model):
        img_y, img_x, _ = img.shape

        image_tensor = self.read_image2(img)
        prediction_mask = self.infer(image_tensor=image_tensor, model=model)
        prediction_colormap = self.decode_segmentation_masks(prediction_mask, colormap, 3)
        overlay = self.get_overlay(image_tensor, prediction_colormap)

        prediction_colormap = self.changeBGR(prediction_colormap, (img_x, img_y))
        overlay = self.changeBGR(overlay, (img_x, img_y))

        return overlay, prediction_colormap

    def changeBGR(self,image, size):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size))
        return image

    def gethairsegement(self, file_name):

        ff = np.fromfile(file_name, np.uint8)
        img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)
        overlay, prediction_colormap = self.plot_predictions2(img, colormap, self.model2)
        return overlay, prediction_colormap
class post_processing:
    def __init__(self, src):
        self.img = src
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def threshold(self):
        result = cv2.adaptiveThreshold(self.img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,2)

    def erosion_img(self):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        erosion = cv2.erode(self.img_gray, k)
        merged = np.hstack((self.img_gray, erosion))

    def k_means(self, resize = False):
        height, width = self.img_gray.shape
        small_img = cv2.resize(self.img_gray, dsize = (0,0), fx = 0.5, fy = 0.5 , interpolation= cv2.INTER_AREA)

        K = 3
        if resize:
            data = small_img.reshape((-1, 1)).astype(np.float32)
        else:
            data = self.img_gray.reshape((-1, 1)).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        # print(center)
        res = center[label.flatten()]
        if resize:
            res = res.reshape((small_img.shape))
        else:
            res = res.reshape((self.img_gray.shape))

        res = cv2.resize(res, (width, height),interpolation= cv2.INTER_CUBIC)

        center = center.tolist()
        center = sum(center, [])

        return res, center

    def contour_process(self, img, center, join_ratio = 0.85):

        black = min(center)
        ret, imthres = cv2.threshold(img, black, 255, cv2.THRESH_BINARY)
        back_img = cv2.cvtColor(imthres, cv2.COLOR_GRAY2BGR)
        back_img2 = cv2.cvtColor(imthres, cv2.COLOR_GRAY2BGR)

        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        dst = cv2.morphologyEx(imthres, cv2.MORPH_CLOSE, k)

        contour, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(back_img, contour, -1, (0, 255, 0), 4)
        sizes = []
        for i in contour:
            for j in i:
                cv2.circle(back_img, tuple(j[0]), 1, (255, 0, 0), -1)
            sizes.append(len(i))

        max_size = max(sizes)
        areas = []
        for i in contour:
            if len(i) < max_size * join_ratio:
                # print(i.reshape(len(i),2))
                cv2.fillPoly(back_img2, [i.reshape(len(i),2)], (0,0,0))
            else:
                temp = i.reshape(len(i),2)
                max_x, max_y = temp.max(axis = 0)
                min_x, min_y = temp.min(axis=0)
                # print(max_x, min_x, max_y, min_y)
                areas.append([min_x, max_x, min_y, max_y])

        areas = np.array(areas)
        t_min_x = min(areas[:,0])
        t_max_x = max(areas[:,1])
        t_min_y = min(areas[:,2])
        t_max_y = max(areas[:,3])

        # cv2.rectangle(back_img2, (t_min_x, t_min_y), (t_max_x, t_max_y), (255,0,0), 1)
        # back_img2 = back_img2[t_min_y:t_max_y+1, t_min_x:t_max_x+1]
        back_img2 = cv2.cvtColor(back_img2, cv2.COLOR_BGR2GRAY)

        return back_img2

    def shift_center(self, img, img_size = 0):
        # print(img.shape)
        i_height, i_width = img.shape
        blcok_size = None
        if i_height >= i_width * 1.5 :
            blcok_size = i_height // 3
        else:
            blcok_size = i_width // 2

        res = np.zeros((blcok_size*4, blcok_size*4), np.uint8)
        res[blcok_size:blcok_size+i_height, blcok_size:blcok_size+i_width] = img

        if img_size != 0:
            res = cv2.resize(res, (img_size, img_size))

        return res

    def post_process(self):
        res, center = self.k_means()
        res = self.contour_process(res, center)

        return res