from django.conf.urls import include
from django.urls import re_path
from . import views
from rest_framework import routers

routers = routers.DefaultRouter()
routers.register('ojmessage', views.OJMessageView)
routers.register('blog', views.BlogView)
routers.register('banner', views.BannerView)

urlpatterns = [
    re_path('', include(routers.urls)),
]
