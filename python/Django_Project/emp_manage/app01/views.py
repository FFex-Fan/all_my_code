from django.shortcuts import render, redirect, HttpResponse
from app01 import models


# Create your views here.

# 部门列表
def depart_list(request):
    """ 部门列表 """

    # 在数据库中，获取所有部门列表
    queryset = models.Department.objects.all()
    return render(request, "depart_list.html",
                  {"queryset": queryset})


def depart_add(request):
    """ 添加部门 """
    if request.method == "GET":
        return render(request, "depart_add.html")
    title = request.POST.get("title")
    models.Department.objects.create(title=title)
    return redirect("/depart/list/")


def depart_delete(request):
    """ 删除部门 """
    no = request.GET.get("depart_id")
    models.Department.objects.filter(id=no).delete()
    return redirect("/depart/list/")


def depart_edit(request, depart_id):
    if request.method == "GET":
        content = models.Department.objects.filter(id=depart_id).first()
        return render(request, "depart_edit.html"
                      , {"content": content})

    new_title = request.POST.get("title")
    models.Department.objects.filter(id=depart_id).update(title=new_title)
    return redirect("/depart/list/")



def user_list(request):
    """ 用户列表 """
    data = models.UserInfo.objects.all()
    # for item in data:
    #     print(item.create_time.strftime("%Y-%m-%d"))
    #     print(item.get_gender_display()) #
    #     print(item.depart_id, item.depart.title)
    return render(request, "user_list.html",
                  {
                      "data": data,
                  })
