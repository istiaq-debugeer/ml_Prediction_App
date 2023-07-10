from django.urls import path
from .import views

urlpatterns = [
    path('',views.dashboard,name='home'),
    path('ml_project_result/', views.ml_project_result, name='ml_project_result'),
    path('document/', views.document, name='document'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('input/', views.input, name='input'),
    path('recommend_cellphones/', views.recommend_cellphones, name='recommend_cellphones'),
    # path('table_Document/', views.index, name='Ditails'),
]