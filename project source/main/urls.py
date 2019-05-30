from django.contrib import admin
from django.urls import path, include
from .views import *
urlpatterns = [
    path('', home, name="home"),
#    path('result/<int:pk>', result, name="result"),
    path('result/<int:pk>/<str:lang>', result, name="result"),
    path('edit/<int:pk>', edit, name="edit"),
]
