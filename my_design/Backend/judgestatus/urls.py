from django.conf.urls import include
from django.urls import re_path

from . import views
from rest_framework import routers

routers = routers.DefaultRouter()
routers.register('judgestatus', views.JudgeStatusView)
routers.register('judgestatusput', views.JudgeStatusPutView)
routers.register('judgestatuscode', views.JudgeStatusCodeView)
routers.register('casestatus', views.CaseStatusView)
routers.register('acrank', views.ACRankView)

urlpatterns = [
    re_path('', include(routers.urls)),
    re_path(r'^rejudge', views.RejudgeAPIView.as_view()),
]
