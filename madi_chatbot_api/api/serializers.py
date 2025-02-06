from rest_framework import serializers
from .models import Intent

class IntentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Intent
        fields = ('__all__')
    
    tag = serializers.CharField(max_length=256)
    patterns = serializers.ListField(child=serializers.CharField(max_length=256))
    responses = serializers.ListField(child=serializers.CharField(max_length=256))