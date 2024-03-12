from django.conf.urls import include
from django.urls import re_path

from . import views
from rest_framework import routers

routers = routers.DefaultRouter()
routers.register('classes', views.ClassDataView)
routers.register('classStudent', views.ClassStudentDataView)

urlpatterns = [
    re_path('', include(routers.urls)),
    re_path(r'^ADDclasses', views.ClassDataAPIView.as_view()),
    re_path(r'^AddClass', views.ClassStudentDataAPIView.as_view()),
    re_path(r'^DeleteClass', views.DeleteClassDataAPIView.as_view()),
]
