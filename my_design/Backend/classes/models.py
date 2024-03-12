from django.db import models


class ClassStudentData(models.Model):
    studentUserName = models.CharField(max_length=50, null=False, default="")
    studentNumber = models.CharField(max_length=50, null=False)
    className = models.CharField(max_length=50, null=False)
    studentRealName = models.CharField(max_length=50, null=False)

    objects = models.Manager()

    def __str__(self):
        return self.studentUserName


class theClasses(models.Model):
    className = models.CharField(max_length=50, default="None")
    classSize = models.CharField(max_length=50, default="None")
    canjoinclass = models.CharField(max_length=50, default="open")

    objects = models.Manager()

    def __str__(self):
        return self.className
