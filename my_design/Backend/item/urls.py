from django.conf.urls import include
from django.urls import re_path

from . import views
from rest_framework import routers

routers = routers.DefaultRouter()
routers.register('putitem', views.ItemPutView)

urlpatterns = [
    re_path('', include(routers.urls)),
    re_path(r'^item', views.ItemGetAPIView.as_view()),
]
