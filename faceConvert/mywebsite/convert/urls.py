from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from . import views
from django.views.static import serve
from django.urls import re_path
from django.conf.urls import url, handler400, handler404, handler500

app_name = 'convert'
urlpatterns = [
    path("",views.index, name="index"),
    path("<int:post_id>/success",views.success, name="success"),
    path("<int:post_id>/failed", views.failed, name="failed"),
    # re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),  # 이부분 추가!!
] + static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)


