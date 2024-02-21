from django.shortcuts import render, redirect

from app01 import models
from app01.utils.pagination import Pagination
from app01.utils.form import MobileForm, MobileEditForm, UserModelForm

def mobile_list(request):
    data_dict = {}
    val = request.GET.get("search", "")
    if val:
        data_dict["mobile__contains"] = val

    data = models.VIPMobile.objects.filter(**data_dict).order_by("-level")  # select * from vipmobile order by level desc;
    page_object = Pagination(request, data)

    context = {
        "search_val": val,
        "list": page_object.page_queryset,  # 分完页的数据
        "page_string": page_object.html()  # 页码
    }

    return render(request, 'mobile_list.html', context)


def mobile_add(request):
    ''' 添加手机号 '''
    if request.method == "GET":
        form = MobileForm()
        return render(request, 'mobile_add.html', {
            "form": form,
        })
    form = MobileForm(request.POST)
    if form.is_valid():
        form.save()
        return redirect("/mobile/list/")
    return render(request, "mobile_add.html", {
        "form": form,
    })




def mobile_edit(request, mobile_id):

    ''' 编辑手机号 '''
    data = models.VIPMobile.objects.filter(id = mobile_id).first()
    if request.method == "GET":
        form = MobileEditForm(instance=data) # 编辑页面显示已经保存的数据
        return render(request, 'mobile_edit.html', {
            "form": form
        })

    form = MobileEditForm(request.POST, instance=data)
    if form.is_valid():
        form.save()
        return redirect("/mobile/list/")
    return render(request, "mobile_edit.html", {
        "form": form,
    })

def mobile_delete(request, mobile_id):
    models.VIPMobile.objects.filter(id = mobile_id).delete()
    return redirect("/mobile/list/")