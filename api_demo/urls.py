# my_app/urls.py
from django.urls import path
from api_demo.views import PredictView

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
]
