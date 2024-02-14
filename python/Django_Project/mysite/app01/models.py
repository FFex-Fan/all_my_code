from django.db import models


# Create your models here.

# 表一
class UserInfo(models.Model):  # UserInfo 类，继承自 models.Model
    name = models.CharField(max_length=32)  # name 为字符串类型
    password = models.CharField(max_length=64)
    age = models.IntegerField(default=18)  # age 为整型
    data = models.IntegerField(null=True, blank=True)
    """ 相当于 SQL 语句：
    create table app01_userinfo(
        id bigint auto_increment primary key, // 自动加上
        name varchar(32),
        password varchar(64),
        age int
    )
    """


# 表二
class Department(models.Model):
    title = models.CharField(max_length=16)


# 表三
class Role(models.Model):
    caption = models.CharField(max_length=16)