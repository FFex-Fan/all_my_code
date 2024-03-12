from django.conf.urls import  include
from django.urls import re_path

from rest_framework import routers
from . import views

routers = routers.DefaultRouter()
routers.register('wiki', views.WikiView)
routers.register('wikicount', views.WikiCountView)
routers.register('mbcode', views.MBCodeView)
routers.register('mbcodedetail', views.MBCodeDetailView)
routers.register('mbcodedetailnocode', views.MBCodeDetailNoCodeView)
routers.register('trainning', views.TrainningContestView)

urlpatterns = [
    re_path('', include(routers.urls)),
]
