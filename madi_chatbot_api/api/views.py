from rest_framework import generics

from .models import Intent
from .serializers import IntentSerializer


class IntentList(generics.ListCreateAPIView):
    serializer_class = IntentSerializer
    
    def get_queryset(self):
        queryset = Intent.objects.values("tag", "patterns", "responses")
        return queryset
    