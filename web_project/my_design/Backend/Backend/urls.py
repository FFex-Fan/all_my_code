from django.contrib import admin
from django.urls import include
from rest_framework.documentation import include_docs_urls
from django.urls import path, re_path
from drf_yasg.views import get_schema_view
from drf_yasg import openapi


SchemaView = get_schema_view(
    openapi.Info(
        title="API 文档接口",
        default_version='v1.0',
        description="""
            API 遵循 REST 标准进行设计。
        """,
        contact=openapi.Contact(email="123123@qq.com"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,

    # permission_classes=(permissions.AllowAny,),  # schema view本身的权限类
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('app01/', include('app01.urls')),
    re_path(r'^docs/', include_docs_urls(title='My api title')),
    re_path(r'^swagger(?P<format>.json|.yaml)$', SchemaView.without_ui(cache_timeout=0), name='schema-json'),  # 导出
    re_path(r'^redoc/$', SchemaView.with_ui('redoc', cache_timeout=0), name='schema-redoc'),  # redoc美化UI
    re_path(r'^swagger/$', SchemaView.with_ui('swagger', cache_timeout=None), name='cschema-swagger-ui'),
    path('', include('problem.urls')),
    path('', include('judgestatus.urls')),
    path('', include('user.urls')),
    path('', include('contest.urls')),
    path('', include('board.urls')),
    path('', include('blog.urls')),
    path('', include('item.urls')),
    path('', include('classes.urls')),
]
