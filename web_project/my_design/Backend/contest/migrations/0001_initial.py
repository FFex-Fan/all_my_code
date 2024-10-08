# Generated by Django 4.2 on 2024-03-09 13:31

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ContestAnnouncement',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('contestid', models.IntegerField()),
                ('announcement', models.CharField(max_length=500)),
            ],
        ),
        migrations.CreateModel(
            name='ContestBoard',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('contestid', models.IntegerField()),
                ('username', models.CharField(max_length=50)),
                ('user', models.CharField(max_length=50)),
                ('problemrank', models.IntegerField()),
                ('type', models.IntegerField()),
                ('submittime', models.BigIntegerField()),
                ('submitid', models.IntegerField()),
                ('rating', models.IntegerField(default=1500)),
            ],
        ),
        migrations.CreateModel(
            name='ContestBoardTotal',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user', models.CharField(max_length=100)),
                ('nickname', models.CharField(max_length=100)),
                ('contestid', models.IntegerField()),
                ('score', models.IntegerField()),
                ('time', models.CharField(max_length=100)),
                ('detail', models.CharField(max_length=500)),
            ],
        ),
        migrations.CreateModel(
            name='ContestChoiceProblem',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ContestId', models.IntegerField()),
                ('ChoiceProblemId', models.CharField(max_length=50)),
                ('rank', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='ContestComingInfo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ojName', models.CharField(max_length=100)),
                ('link', models.CharField(max_length=200)),
                ('startTime', models.BigIntegerField()),
                ('endTime', models.BigIntegerField()),
                ('contestName', models.CharField(max_length=500)),
            ],
        ),
        migrations.CreateModel(
            name='ContestComment',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('contestid', models.IntegerField()),
                ('user', models.CharField(max_length=50)),
                ('title', models.CharField(default='提问', max_length=50)),
                ('problem', models.CharField(default='ALL', max_length=500)),
                ('message', models.CharField(max_length=500)),
                ('huifu', models.CharField(default='No respones', max_length=500)),
                ('time', models.DateTimeField(auto_now=True)),
                ('rating', models.IntegerField(default=1500)),
            ],
        ),
        migrations.CreateModel(
            name='ContestInfo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('creator', models.CharField(default='admin', max_length=50)),
                ('oj', models.CharField(default='LPOJ', max_length=50)),
                ('title', models.CharField(default='contest', max_length=50)),
                ('level', models.IntegerField(default=1)),
                ('des', models.CharField(default='contest des', max_length=500)),
                ('note', models.CharField(default='contest note', max_length=500)),
                ('begintime', models.DateTimeField()),
                ('lasttime', models.IntegerField(default=18000)),
                ('type', models.CharField(default='ACM', max_length=50)),
                ('auth', models.IntegerField(default=2)),
                ('clonefrom', models.IntegerField(default=-1)),
                ('classes', models.CharField(default='All', max_length=500)),
                ('iprange', models.CharField(default='iprange', max_length=2000)),
                ('lockboard', models.IntegerField(default=0)),
                ('locktime', models.IntegerField(default=0)),
            ],
        ),
        migrations.CreateModel(
            name='ContestProblem',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('contestid', models.IntegerField()),
                ('problemid', models.CharField(max_length=50)),
                ('problemtitle', models.CharField(default='uname', max_length=500)),
                ('rank', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='ContestRatingChange',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('contestid', models.IntegerField()),
                ('contestname', models.CharField(max_length=100)),
                ('contesttime', models.CharField(max_length=100)),
                ('user', models.CharField(max_length=50)),
                ('lastrating', models.IntegerField(default=0)),
                ('ratingchange', models.IntegerField(default=0)),
                ('currentrating', models.IntegerField(default=0)),
            ],
        ),
        migrations.CreateModel(
            name='ContestRegister',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('contestid', models.IntegerField()),
                ('user', models.CharField(max_length=50)),
                ('rating', models.IntegerField(default=0)),
            ],
        ),
        migrations.CreateModel(
            name='ContestTutorial',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('contestid', models.IntegerField()),
                ('value', models.TextField(default='暂无数据！')),
            ],
        ),
        migrations.CreateModel(
            name='StudentChoiceAnswer',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(default='', max_length=100)),
                ('realname', models.CharField(default='', max_length=100)),
                ('number', models.CharField(default='', max_length=100)),
                ('contestid', models.CharField(default='', max_length=100)),
                ('answer', models.CharField(max_length=100)),
                ('answer_detail', models.TextField(default='')),
                ('score', models.IntegerField()),
            ],
        ),
    ]
