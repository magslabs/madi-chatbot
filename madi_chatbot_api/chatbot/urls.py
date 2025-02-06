from django.urls import path
from .views import TrainChatbot, PromptChatbot

urlpatterns = [
    path('train', TrainChatbot.as_view(), name='train'),
    path('prompt', PromptChatbot.as_view(), name='prompt'),
]
