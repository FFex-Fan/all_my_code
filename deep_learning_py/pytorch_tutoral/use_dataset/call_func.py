# coding=utf-8
class Person:
    def __call__(self, name):
        print("__call__" + name)

    def hello(self, name):
        print("hello" + name)

person = Person()
person("zhangsan") # 可以使用内置对象来调用函数
person.hello("list") # 使用 . 调用函数