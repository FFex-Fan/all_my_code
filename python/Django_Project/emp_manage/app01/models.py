from django.db import models


# Create your models here.

class Department(models.Model):
    """ 部门表 """
    title = models.CharField(max_length=32, verbose_name='标题')

    def __str__(self):  # 返回字段重写为部门名称（类似于重写 to_string）
        return self.title


class UserInfo(models.Model):
    """ 员工表 """
    name = models.CharField(max_length=16, verbose_name='姓名')
    password = models.CharField(max_length=16, verbose_name='密码')
    age = models.IntegerField(verbose_name='年龄')
    email = models.EmailField(verbose_name='电子邮件', null=True, blank=True)
    # DecimalField 中，总长度为10，小数占2, 默认为 0
    account = models.DecimalField(max_digits=10, decimal_places=2, default=0, verbose_name='账户余额')
    create_time = models.DateTimeField(verbose_name='入职时间')

    # 无约束
    # depart_id = models.BigIntegerField(verbose_name='部门ID')

    """
    1. 有约束
        to: 与哪张表关联
        to_field: 与那个列关联
    2. Django 自动
        - 写的 depart
        - 生成的数据列 depart_id
    3. 部门表被删除
        - 级联删除：depart = models.ForeignKey(to='Department',to_field='id', on_delete=models.CASCADE)
        - 置空：depart = models.ForeignKey(to='Department', to_field='id', null=True, blank=True, on_delete=models.SET_NULL())
    """
    depart = models.ForeignKey(verbose_name='部门', to='Department', to_field='id', on_delete=models.CASCADE)

    # 在 Django 中做的约束，只能选 1 或 2，通过 1 / 2 对应到 男 / 女
    gender_choices = (
        (1, '男'),
        (2, '女'),
    )
    gender = models.SmallIntegerField(verbose_name='性别', choices=gender_choices)
