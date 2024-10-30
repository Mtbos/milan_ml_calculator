"""
URL configuration for ml_calculator project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from ml_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('ml_app.urls')),  # Include URLs from ml_app
    path('', views.home, name='home'),  # Public homepage
    path('signup/', views.signup, name='signup'),  # Signup page
    path('accounts/', include('django.contrib.auth.urls')),  # Login and logout URLs
    path('logout/', views.logout_view, name='logout'),
    
]

