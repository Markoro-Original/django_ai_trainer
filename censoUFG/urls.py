from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.ia_import, name='ia_import'),
    path('ia_import', views.ia_import, name='ia_import'),
    path('ia_import_save', views.ia_import_save, name='ia_import_save'),
    path('ia_import_list', views.ia_import_list, name='ia_import_list'),
    path('ia_knn_treino', views.ia_knn_treino, name='ia_knn_treino'),
    path('ia_knn_matriz', views.ia_knn_matriz, name='ia_knn_matriz')
]