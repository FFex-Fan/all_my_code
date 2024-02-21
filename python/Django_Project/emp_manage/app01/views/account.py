
from django import forms
from django.shortcuts import render, redirect

class LoginForm(forms.Form):
    username = forms.CharField(
        label='用户名',
        widget=forms.TextInput(
            attrs={
                'class': 'form-control',
                'placeholder': '用户名'
            }
        )
    )
    password = forms.CharField(
        label='密码',
        widget=forms.PasswordInput(
            attrs={
                'class': 'form-control',
                'placeholder': '密码'
            }
        )
    )



def login(request):
    form = LoginForm()
    return render(request, 'login.html', {
        'form': form
    })