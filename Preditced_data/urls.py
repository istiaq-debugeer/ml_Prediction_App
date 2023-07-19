from django.urls import path
from .import views

urlpatterns = [
    path('',views.SignupPage,name='signup'),
    path('login/',views.LoginPage,name='login'),
    path('logout/',views.LogoutPage,name='logout'),

    path('ml_project_result/', views.ml_project_result, name='ml_project_result'),
    path('document/', views.document, name='document'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('input/', views.input, name='input'),
    path('recommend_cellphones/', views.recommend_cellphones, name='recommend_cellphones'),
    path('ditails/', views.index, name='index'),
]