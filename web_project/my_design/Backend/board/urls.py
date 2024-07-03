from django.conf.urls import include
from django.urls import re_path

from . import views
from rest_framework import routers

routers = routers.DefaultRouter()
routers.register('board', views.BoardView)
routers.register('dailyboard', views.DailyBoardView)
routers.register('teamboard', views.TeamBoardView)
routers.register('dailycontestboard', views.DailyContestBoardView)
routers.register('settingboard', views.SettingBoardView)

urlpatterns = [
    re_path('', include(routers.urls)),
]
