import os
import re

import cv2
from PIL import Image
import socket


HOST = '127.0.0.1'
PORT = 9999

#django
from .models import Post
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import redirect, get_object_or_404
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import HttpResponseRedirect
from django.urls import reverse

#source
from .source.dlib import get_people_num
from .source.hair_detech import removeBack, hairsegement, post_processing
from .source.hair_similarity import similarity
from .source.get_hair_json import get_hair_info


ai = hairsegement()
rm = removeBack()
sim = similarity()
media_dir = './static/object/data/3D/'
image_dir = './media/'
hair_info = get_hair_info()

# Create your views here.
@csrf_exempt
def index(request):
    if request.method == "POST":
        post = Post()
        post.name = request.POST['name']
        try:
            post.image = request.FILES['image']
        except:
            post.image = None
            print("image None")

        # print(post.name)
        # print(request.FILES['image'], type(request.FILES['image']))
        # print(post.image, type(post.image), post.image.name)
        post.save()
        # print(post.id)
        # print("postimagename",post.image.name)

        if post.image != None :
            file_name = image_dir + post.image.name
            nums = get_people_num(file_name)
            print('People in image nums', nums)
            if nums == 1:

                create_Post(post.id)

                return redirect(str(post.id)+'/success')

            else:
                return redirect(str(post.id) + '/failed')
        else:

            return redirect(str(post.id)+'/failed')
    else:
        front_dir = media_dir + 'front'
        front_files = os.listdir(front_dir)
        base_files = os.listdir(media_dir + 'back')
        temp = []
        for base in base_files:
            temp.append(re.sub(r'[^0-9]', '', base))

        context = {
            "front_dir": front_dir, "front_files": front_files, "media_dir": media_dir, "base_files": temp
        }
        return render(request, "main.html", context=context)

@csrf_exempt
def success(request, post_id):
    #delete
    if request.method == "POST":
        post = get_object_or_404(Post, pk=post_id)

        upper_dir = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

        os.remove(upper_dir+"/"+post.hair_mask_url)
        os.remove(upper_dir+"/"+post.hair_area_url)
        os.remove(upper_dir+"/"+post.face_3D_url)
        os.remove(upper_dir+"/media/" + post.image.name)

        post.delete()

        return redirect('/')

    else:
        post = get_object_or_404(Post, pk=post_id)

        front_dir = media_dir + 'front'
        front_files = os.listdir(front_dir)
        base_files = os.listdir(media_dir + 'back')
        temp = []
        for base in base_files:
            temp.append(re.sub(r'[^0-9]', '', base))

        context = {
            "front_dir": front_dir, "front_files": front_files, "media_dir": media_dir, "base_files": temp, 'post' : post, "hair_info":hair_info
        }

        return render(request, "success.html", context=context)


@csrf_exempt
def failed(request, post_id):
    if request.method == "POST":
        post = get_object_or_404(Post, pk=post_id)
        post.name = request.POST['name']

        upper_dir = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

        os.remove(upper_dir+"/media/" + post.image.name)
        try:
            post.image = request.FILES['image']
        except:
            post.image = None
            print("image None")

        # print(post.name)
        # # print(request.FILES['image'], type(request.FILES['image']))
        # print(post.image, type(post.image), post.image.name)
        post.save()
        # print(post.id)
        # print("postimagename", post.image.name)

        if post.image != None:
            file_name = image_dir + post.image.name
            nums = get_people_num(file_name)
            print('People in image nums,', nums)
            if nums == 1:

                create_Post(post.id)

                return HttpResponseRedirect(reverse('convert:success', args={post_id,}))

            else:
                return HttpResponseRedirect(reverse('convert:failed', args={post_id,}))

        else:
            return HttpResponseRedirect(reverse('convert:failed', args={post_id,}))

    else:
        post = get_object_or_404(Post, pk=post_id)

        context = {
            'post': post
        }

        return render(request, "failed.html", context=context)


def create_Post(post_id):

    post = get_object_or_404(Post, pk=post_id)

    file_name = image_dir + post.image.name


    # 소켓프로그램으로 3D객체 생성
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    client_socket.sendall(file_name.encode())
    data = client_socket.recv(1024)
    data = str(data.decode())
    # print('Received', data)
    client_socket.close()


    mask = removeBack().getremovemask(file_name)
    overlay, prediction_colormap = hairsegement().gethairsegement(file_name)

    overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
    prediction_colormap = cv2.bitwise_and(prediction_colormap, overlay, mask=mask)

    res = post_processing(prediction_colormap).post_process()

    similar_list = sim.get_similar_hair_list(res)
    # print(similar_list[:3])

    # convert PIL
    overlay_converted = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    pil_overlay = Image.fromarray(overlay_converted)
    mask_converted = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    pil_mask = Image.fromarray(mask_converted)

    image_name = post.image.name.rsplit('.')[0]
    image_name = image_name.rsplit('/')[1]

    area_url = 'media/areas/%s.png' % image_name
    mask_url = 'media/masks/%s.png' % image_name
    pil_overlay.save(area_url, 'png')
    pil_mask.save(mask_url, 'png')

    post.hair_area_url = area_url
    post.hair_mask_url = mask_url

    post.ssim1_name = similar_list[0]['name']
    post.ssim2_name = similar_list[1]['name']
    post.ssim3_name = similar_list[2]['name']

    post.ssim1_url = 'object/data/ssim/' + similar_list[0]['ssim'] + '.png'
    post.ssim2_url = 'object/data/ssim/' + similar_list[1]['ssim'] + '.png'
    post.ssim3_url = 'object/data/ssim/' + similar_list[2]['ssim'] + '.png'

    # print(post.ssim1_name, post.ssim2_name, post.ssim3_name)


    #소켓으로 받은 데이터 정리
    data = data.split('/')
    # print(data)

    post.ratio_x = data[0]
    post.ratio_y = data[1]
    post.ratio_z = data[2]

    post.front_x = data[3]
    post.front_y = data[4]
    post.front_z = data[5]

    post.side_x = data[6]
    post.side_y = data[7]
    post.side_z = data[8]

    post.back_x = data[9]
    post.back_y = data[10]
    post.back_z = data[11]

    save_name = file_name.split('/')
    save_name = save_name[-1]
    save_name = save_name.split('.')
    save_name = save_name[0]
    save_dir = 'media/3D/%s.glb' % save_name

    post.face_3D_url = save_dir

    post.save()