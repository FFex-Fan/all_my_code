from django.shortcuts import render, HttpResponse

# Create your views here.

def index(request):  # request 为默认参数
    return HttpResponse("欢迎使用")

def user_add(request):
    # 若在 setting,py 中的 DIRS 中加入了 os.path.join() 则会优先从根目录的 templates 中开始寻找【不配置就无效】
    # render() 在 app 的目录下的 templates 中寻找 user_add.html 文件(根据 app 的注册顺序，逐一去他们的 templates 目录中找)
    return render(request, "user_add.html")

def user_list(request):
    return render(request, "user_list.html")