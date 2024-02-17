from django.shortcuts import render, HttpResponse, redirect
from app01 import models


# Create your views here.

def index(request):  # request 为默认参数
    return HttpResponse("欢迎使用")


def user_add(request):
    # 若在 setting,py 中的 DIRS 中加入了 os.path.join() 则会优先从根目录的 templates 中开始寻找【不配置就无效】
    # render() 在 app 的目录下的 templates 中寻找 user_add.html 文件(根据 app 的注册顺序，逐一去他们的 templates 目录中找)
    return render(request, "user_add.html")


def user_list(request):
    return render(request, "user_list.html")


def tpl(request):
    name = 'fzy'
    list = ['a', '1', 'cool']
    user_info = {
        'name': 'John',
        'salary': 10000,
    }

    # 将 name 变量传到 tpl.html 中
    return render(request, 'tpl.html', {
        'name': name,
        'list': list,
        'user_info': user_info,
    })


def something(request):  # request 是一个对象，封装了用户通过浏览器发送过来的所有 请求相关的数据

    # 1. 获取请求方式 GET / POST
    # print(request.method)

    # 2. 在 URL 上传递值
    # print(request.GET)

    # 3. 在请求体中提交数据（数据存放在 HTTP body 中）
    # print(request.POST)

    # 4. 【响应】HttpResponse() 将内容返回给请求者
    # return HttpResponse("返回内容")

    # 5. 【响应】读取 HTML 内容 + 渲染 => 字符串，返回给用户浏览器
    # return render(request, 'something.html')

    # 6. 【响应】重定向到其他页面
    return redirect("https://www.baidu.com")


def login(request):
    """
        1. 首次访问时，以 GET 方式获取 login.html
        2. 在提交表单信息后，通过 POST 方式访问 login.html

    """
    if request.method == "GET":
        return render(request, 'login.html')
    # 如果是 POST 请求，获取用户提交的数据
    print(request.POST)
    username = request.POST.get("user")
    pwd = request.POST.get("pwd")

    if username == 'root' and pwd == '123':
        return redirect("https://www.baidu.com")
    else:
        return render(request, 'login.html',
                      {'error_message': "用户名或密码错误"})


def orm(request):
    # 测试 ORM 操作表中的数据
    data_list = models.UserInfo.objects.all()
    for obj in data_list:
        print(obj.id, obj.name, obj.password, obj.data)
    return HttpResponse("成功")


def info_list(request):
    data_list = models.UserInfo.objects.all()
    return render(request, 'info_list.html',{'data_list':data_list})


def info_add(request):
    if request.method == "GET":
        return render(request, 'info_add.html')

    # 获取用户提交的数据
    username = request.POST.get("user")
    password = request.POST.get("pwd")
    age = request.POST.get("age")

    # 添加到数据库
    models.UserInfo.objects.create(name=username, password=password, age=age)
    # 重定向到本地某个网页直接写后面的部分即可，会自行拼接域名
    return redirect("/info/list")

def info_delete(request):
    id = request.GET.get("id")
    models.UserInfo.objects.filter(id=id).delete()
    return redirect("/info/list")