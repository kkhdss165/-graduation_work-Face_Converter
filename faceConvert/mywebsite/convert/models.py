from django.db import models

# Create your models here.

class Post(models.Model):
    name = models.CharField(max_length=200)
    image = models.ImageField(upload_to="images/", null=True, blank=True)

    #가공처리 이미지
    hair_area_url = models.CharField(max_length=200, null = True)
    hair_mask_url = models.CharField(max_length=200, null = True)

    #유사 이미지
    ssim1_url = models.CharField(max_length=200, null = True)
    ssim1_name = models.CharField(max_length=200, null = True)
    ssim2_url = models.CharField(max_length=200, null = True)
    ssim2_name = models.CharField(max_length=200, null = True)
    ssim3_url = models.CharField(max_length=200, null = True)
    ssim3_name = models.CharField(max_length=200, null = True)

    #3D모델
    face_3D_url = models.CharField(max_length=200, null = True)

    #3D모델 좌표
    front_x = models.FloatField(null = True)
    front_y = models.FloatField(null=True)
    front_z = models.FloatField(null=True)

    side_x = models.FloatField(null=True)
    side_y = models.FloatField(null=True)
    side_z = models.FloatField(null=True)

    back_x = models.FloatField(null=True)
    back_y = models.FloatField(null=True)
    back_z = models.FloatField(null=True)

    #비율
    ratio_x = models.FloatField(null=True)
    ratio_y = models.FloatField(null=True)
    ratio_z = models.FloatField(null=True)