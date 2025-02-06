import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import random
import pickle

from rest_framework import generics
from rest_framework.response import Response

from chatbot import services as chatbotService

from .serializers import InitializeChatbotSerializer, PromptChatbotSerializer, TrainChatbotSerializer

data = []
words = []
classes = []

lemmatizer = WordNetLemmatizer()

# Load trained model, words, and classes
def initialize_chatbot():
    global data, model, words, classes
    
    try:
        data = chatbotService.load_data()
        model = chatbotService.load_trained_model()
        words = chatbotService.load_words()
        classes = chatbotService.load_classes()
        
        chatbotService.setup_nltk()
        
        print("Chatbot initialized successfully.")
    except FileNotFoundError as e:
        print("Chatbot initialization failed. Please train the chatbot first.")


def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list
    
def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    for i in intents_json["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])
    return "I'm sorry, I do not understand your question."

class InitializeChatbot(generics.ListAPIView):
    serialize_class = InitializeChatbotSerializer
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        initialize_chatbot()
        
    def get(self, request):
        response = {"response": "Chatbot has been initialized successfully."}
        serializer_class = InitializeChatbotSerializer(response)
        return Response(serializer_class.data)

class TrainChatbot(generics.ListAPIView):
    serializer_class = TrainChatbotSerializer
    def get(self, request):
        chatbotService.start_training_process()
        
        response = {"status": "success", "message": "Chatbot has been trained successfully."}
        serializer_class = TrainChatbotSerializer(response)
        return Response(serializer_class.data)

class PromptChatbot(generics.CreateAPIView):
    serializer_class = PromptChatbotSerializer
    
    def create(self, request, *args, **kwargs):
        # Extract the prompt from the incoming request data
        message = request.data.get("prompt")
        if not message:
            return Response({"error": "No prompt provided"}, status=400)

        try:
            # Predict class and generate a response
            ints = predict_class(message, model)
            response = get_response(ints, data)
        except Exception as e:
            response = "I'm sorry, I do not understand your question."

        # Prepare response data to return
        response_data = {
            "prompt": message,
            "response": response
        }
        
        # Serialize the response data using the serializer
        serializer = PromptChatbotSerializer(response_data)

        return Response(serializer.data, status=201)