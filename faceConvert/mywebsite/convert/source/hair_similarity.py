import cv2
import os
import numpy as np
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
import json
from operator import itemgetter

file_path = "./static/object/data/info.json"

class similarity():

    def __init__(self):
        with open(file_path, 'r') as file:
            data = json.load(file)
            # print(type(data))
            # print(data)
            self.models = data['list']

    def get_mask(self, file_name):
        img = cv2.imread(file_name)
        target = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(target,1,255, cv2.THRESH_BINARY)

        contour2, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x = []
        y = []
        for i in contour2:
            if len(i) > 5 :
                for j in i:
                    x.append(j[0][0])
                    y.append(j[0][1])

        # img = cv2.rectangle(img, (min(x), min(y)), (max(x), max(y)), (255,0,0), 1)
        target_mask = thresh1[min(y):max(y)+1, min(x):max(x)+1]

        return target_mask

    def get_mask_img(self, img):

        ret, thresh1 = cv2.threshold(img,1,255, cv2.THRESH_BINARY)

        contour2, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x = []
        y = []
        for i in contour2:
            if len(i) > 5 :
                for j in i:
                    x.append(j[0][0])
                    y.append(j[0][1])

        # img = cv2.rectangle(img, (min(x), min(y)), (max(x), max(y)), (255,0,0), 1)
        target_mask = thresh1[min(y):max(y)+1, min(x):max(x)+1]

        return target_mask

    def get_similar_hair_list(self,target_img, block_nums = 12):

        total = []

        target_mask = self.get_mask_img(target_img)
        sum = 0
        for model in self.models:
            file_name = './static/object/data/ssim/' + model['ssim'] +'.png'
            print(file_name)

            num = 1
            if model['reverse'] : num += 1

            for i in range(num):

                save_name = model['name']
                loop_mask = self.get_mask(file_name)

                if i == 1:
                    save_name = save_name + 'r'
                    loop_mask = cv2.flip(loop_mask,1)

                t_h, t_w = target_mask.shape
                l_h, l_w = loop_mask.shape

                # 너비 길이 맞추고 조합
                min_w = min(t_w, l_w)
                target_mask = cv2.resize(target_mask, dsize=(0, 0), fx=(min_w / t_w), fy=(min_w / t_w))
                loop_mask = cv2.resize(loop_mask, dsize=(0, 0), fx=(min_w / l_w), fy=(min_w / l_w))

                t_h, t_w = target_mask.shape
                l_h, l_w = loop_mask.shape

                re_h = t_h + l_h
                re_w = t_w + l_w

                new_canvas_loop = np.zeros((re_h, re_w), np.uint8)

                new_canvas_loop[re_h // 2 - l_h // 2: re_h // 2 - l_h // 2 + l_h,
                re_w // 2 - l_w // 2: re_w // 2 - l_w // 2 + l_w] = loop_mask

                maxscore = None
                min_h = None
                min_w = None

                block_size = min(re_h - t_h, re_w - t_w) // block_nums

                for height in range(0, re_h - t_h, block_size+1):
                    for width in range(0, re_w - t_w, block_size+1):
                        new_canvas_target = np.zeros((re_h, re_w), np.uint8)
                        new_canvas_target[height:height + t_h, width:width + t_w] = target_mask

                        bit_xor = cv2.bitwise_xor(new_canvas_loop, new_canvas_target)
                        bit_and = cv2.bitwise_and(new_canvas_loop, new_canvas_target)

                        and_sum = np.sum(bit_and) // 255
                        xor_sum = np.sum(bit_xor) // 255

                        score = xor_sum
                        # print(and_sum,xor_sum, score)
                        # print(sum(sum(bit_and)),sum(sum(bit_xor)), sum(sum(bit_and)) - sum(sum(bit_xor)))
                        # cv2.imshow("new_canvas_loop", new_canvas_loop)
                        # cv2.imshow("new_canvas_target", new_canvas_target)
                        #
                        # cv2.imshow("bit_xor", bit_xor)
                        # cv2.imshow("bit_and", bit_and)
                        # cv2.waitKey()

                        if maxscore == None or maxscore > score:
                            maxscore = score
                            min_h = height
                            min_w = width

                new_canvas_target = np.zeros((re_h, re_w), np.uint8)
                new_canvas_target[min_h:min_h + t_h, min_w:min_w + t_w] = target_mask

                bit_xor = cv2.bitwise_xor(new_canvas_loop, new_canvas_target)
                bit_and = cv2.bitwise_and(new_canvas_loop, new_canvas_target)
                bit_or = cv2.bitwise_or(new_canvas_loop, new_canvas_target)
                # cv2.imshow("new_canvas_loop", new_canvas_loop)
                # cv2.imshow("new_canvas_target", new_canvas_target)
                #
                # cv2.imshow("bit_xor", bit_xor)
                # cv2.imshow("bit_and", bit_and)
                # cv2.imshow("bit_or", bit_or)

                # print(file_path, "xor: ",score)

                # score = mse(bit_and, new_canvas_target)
                # score2 = mse(bit_or, new_canvas_target)
                score = (mse(bit_or, bit_and))/255
                score2 = (mse(new_canvas_loop, new_canvas_target))/255
                score3 = score + score2

                print(f"Similarity: {score:.5f}", f"Similarity: {score2:.5f}", f"sum: {score3:.5f}")
                # cv2.waitKey()
                sum = score3

                print(save_name, "sum: ", sum)
                total.append({"name":save_name,"diff":sum, "ssim": model['ssim']})

        total = sorted(total, key=itemgetter('diff'))

        return total

    def get_part(self, model_name):
        is_reverse = False
        front_L = None
        front_R = None
        base = None
        if model_name[-1] == 'r':
            is_reverse = True
            model_name = model_name[:-1]
        for model in self.models:
            if model['name'] == model_name:
                front_R = model['front-R']

                if model['front-L'] == 'same':
                    front_L = front_R
                elif model['front-L'] == 'none':
                    front_L = None
                else:
                    front_L = model['front-L']

                if model['base-set'] == 'none':
                    base = None
                else:
                    base = model['base-set']

                break

        if is_reverse:
            front_R, front_L = front_L, front_R

        return front_R, front_L, base