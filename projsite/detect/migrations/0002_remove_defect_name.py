# Generated by Django 3.2.9 on 2021-11-07 07:30

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('detect', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='defect',
            name='name',
        ),
    ]
