from django.urls import path, include
from .views import IntentList

urlpatterns = [
    path('intents', IntentList.as_view(), name='intents'),
    
    #chatbot routes
    path('chatbot/', include('chatbot.urls')),
]
