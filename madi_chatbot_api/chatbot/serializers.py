from rest_framework import serializers

class InitializeChatbotSerializer(serializers.Serializer):
    response = serializers.CharField(max_length=256)
    
class TrainChatbotSerializer(serializers.Serializer):
    status = serializers.CharField(max_length=256)
    message = serializers.CharField(max_length=256)
    
class PromptChatbotSerializer(serializers.Serializer):
    prompt = serializers.CharField(max_length=1024)
    response = serializers.CharField(max_length=1024)
