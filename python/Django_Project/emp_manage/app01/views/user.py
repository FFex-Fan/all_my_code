from django.shortcuts import render, redirect
from app01 import models

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

def user_add(request):
    """ 添加用户 """
    if request.method == "GET":
        context = {
            'gender_choices': models.UserInfo.gender_choices,
            'depart': models.Department.objects.all()
        }

        return render(request, "user_add.html", context)
    name = request.POST.get("name")
    gender = request.POST.get("gender")
    age = request.POST.get("age")
    email = request.POST.get("email")
    password = request.POST.get("password")
    account = request.POST.get("account")
    create_time = request.POST.get("create_time")
    depart_id = request.POST.get("depart_id")
    models.UserInfo.objects.create(
        name=name,
        gender=gender,
        age=age,
        email=email,
        password=password,
        account=account,
        create_time=create_time,
        depart_id=depart_id,
    )
    return redirect("/user/list/")

# modelForm 方式
from django import forms

class MyForm(forms.ModelForm):
    name = forms.CharField(min_length=2, max_length=10, label='用户名')
    age = forms.IntegerField(max_value=200, min_value=1, label='年龄')

    class Meta:
        model = models.UserInfo
        # fields 中的属性，为 models.py 中定义的属性
        fields = ["name", "gender", "age", "email", "password", "account", "create_time", "depart"]

        # 通过控制插件，进行 CSS 美化
        # widgets = {
        #     "name": forms.TextInput(attrs={"class": "form-control"}),
        #     "password": forms.PasswordInput(attrs={"class": "form-control"}),
        # }

    def __init__(self, *args, **kwargs):  # 重新定义 init 方法，
        super().__init__(*args, **kwargs)
        """ 循环找到所有插件，并定义 class 属性
            name: fields 中的内容
            field: name 对应的对象
            field.widget.attrs: 得到插件对应的属性
        """
        for name, field in self.fields.items():
            field.widget.attrs = {
                "class": "form-control",
            }


def model_form_add(request):
    if request.method == "GET":
        form = MyForm()
        context = {
            "form": form,
        }
        return render(request, 'user_model_form_add.html', context)

    # 数据校验
    form = MyForm(request.POST)
    if form.is_valid():  # 成功
        # 如果数据合法，保留到数据库
        form.save()  # MyForm 中的定义的 UserInfo 表中
        return redirect("/user/list/")

    # 失败 (在页面上显示错误信息)
    return render(request, "user_model_form_add.html", {
        "form": form,
    })


def user_edit(request, user_id):  # 路径中有参数，此处必须要添加参数
    """ 编辑用户 """
    # 根据 id 在数据库中获取要编辑的那一行的数据(对象)
    data = models.UserInfo.objects.filter(id=user_id).first()

    if request.method == "GET":

        form = MyForm(instance=data) # instanca 默认将数据显示到前端页面中

        return render(request, 'user_edit.html', {"form": form})
    form = MyForm(request.POST, instance=data) # 更新时需要指定 instance=data，否则为新增数据

    if form.is_valid():
        # 默认保存的是用户输入的所有数据，如果想要在用户输入以外增加一些额外值：
        # form.instance.字段名 = 值
        form.save()
        return redirect("/user/list/")
    return render(request, "user_edit.html", {
        "form": form,
    })


def user_delete(request, user_id):
    models.UserInfo.objects.filter(id=user_id).delete()
    return redirect("/user/list/")