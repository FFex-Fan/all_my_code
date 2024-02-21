from django.shortcuts import render, redirect
from django.core.exceptions import ValidationError

from app01 import models
from app01.utils.pagination import Pagination
from app01.utils.encrypt import md5

def admin_list(request):
    """ 管理员列表 """

    data_dict = {}
    val = request.GET.get("search", "")
    if val:
        data_dict["username__contains"] = val

    list = models.Admin.objects.filter(**data_dict)

    page_object = Pagination(request, list)
    context = {
        'search_val': val,
        'list': page_object.page_queryset,
        'page_string': page_object.html(),
    }
    return render(request, "admin_list.html", context)

from django import forms
from app01.utils.bootstrap import BootStrapModelForm
class AdminForm(BootStrapModelForm):
    comfirm = forms.CharField( # 数据库中没有该字段，额外添加到 html 中的标签
        label="确认密码",
        widget=forms.PasswordInput(render_value=True), # 定义输入框类型, render_value=True：错误时密码不清空
    )

    class Meta:
        model = models.Admin
        fields = ['username', 'password','comfirm']
        widgets = {
            'password': forms.PasswordInput(render_value=True)
        }

    def clean_password(self):
        pwd = self.cleaned_data.get("password")
        return md5(pwd)

    def clean_comfirm(self):  # 钩子函数
        pwd = self.cleaned_data.get("password")
        print(pwd)
        c_pwd = md5(self.cleaned_data.get("comfirm"))
        print(c_pwd)
        if pwd != c_pwd:
            raise ValidationError("密码不匹配")
        return c_pwd


def admin_add(request):
    """ 添加管理员 """
    title = '添加管理员'

    if request.method == 'GET':
        form = AdminForm()
        return render(request, 'change.html', {
            'title': title,
            'form': form,
        })
    form = AdminForm(request.POST)
    if form.is_valid():
        form.save()
        # print(form.cleaned_data)
        return redirect('/admin/list/')
    return render(request, 'change.html', {
        'form': form,
        'title': title,
    })