from django.urls import path
from .views import InitializeChatbot, TrainChatbot, PromptChatbot

urlpatterns = [
    path('initialize', InitializeChatbot.as_view(), name="initialize"),
    path('train', TrainChatbot.as_view(), name='train'),
    path('prompt', PromptChatbot.as_view(), name='prompt'),
]
