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

from .serializers import PromptChatbotSerializer, TrainChatbotSerializer, ChatbotCustomResponseSerializer

data = []
words = []
classes = []
documents = []
ignore_words = ["?", "!"]
lemmatizer = WordNetLemmatizer()

train_x = []
train_y = []

# Load data from JSON file
def load_data():
    global data
    with open('data.json', 'r') as file:
        data = json.load(file)
        
def load_trained_model():
    if os.path.exists("chatbot_model.h5"):
        return load_model("chatbot_model.h5")
    else:
        raise FileNotFoundError("Model file 'chatbot_model.h5' not found. Please train the chatbot first.")
    
def load_words():
    if os.path.exists("words.pkl"):
        return pickle.load(open("words.pkl", "rb"))
    else:
        raise FileNotFoundError("Words file 'words.pkl' not found. Please train the chatbot first.")
    
def load_classes():
    if os.path.exists("classes.pkl"):
        return pickle.load(open("classes.pkl", "rb"))
    else:
        raise FileNotFoundError("Classes file 'classes.pkl' not found. Please train the chatbot first.")
        
# Step 1: Dataset
def setup_nltk():
    nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    
    if not os.path.exists(os.path.join(nltk_data_path, "tokenizers/punkt")):
        nltk.download("punkt", download_dir=nltk_data_path)
    if not os.path.exists(os.path.join(nltk_data_path, "corpora/wordnet")):
        nltk.download("wordnet", download_dir=nltk_data_path)
    if not os.path.exists(os.path.join(nltk_data_path, "tokenizers/punkt_tab")):
        nltk.download('punkt_tab', download_dir=nltk_data_path)
        
    load_data()

def setup_preprocessing_of_data():
    # Step 2: Preprocessing
    def preprocess_data():
        global words, classes, documents, lemmatizer
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                word_list = word_tokenize(pattern)
                words.extend(word_list)
                documents.append((word_list, intent["tag"]))
                if intent["tag"] not in classes:
                    classes.append(intent["tag"])

        words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
        words = sorted(set(words))
        classes = sorted(set(classes))
        
    # Step 3: Save Preprocessed Data
    def save_preprocessed_data():
        # Save words and classes
        pickle.dump(words, open("words.pkl", "wb"))
        pickle.dump(classes, open("classes.pkl", "wb"))
        
    preprocess_data()
    save_preprocessed_data()
    
# Step 4: Create Training Data
def setup_training_data():
    global train_x, train_y, documents
    
    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        word_patterns = doc[0]
        word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
        for w in words:
            bag.append(1) if w in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))

def initialize_training_data():
    setup_nltk()
    setup_preprocessing_of_data()
    setup_training_data()
    
# initialize_training_data()
    
# Step 5: Build Model
def build_model():
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation="softmax"))

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    
    return model

# Step 6: Train Model
def train_model():
    model = build_model()
    model.fit(train_x, train_y, epochs=500, batch_size=8, verbose=1)
    model.save("chatbot_model.h5")

def clean_up_sentence(sentence):
    global lemmatizer
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
        
class TrainChatbot(generics.ListAPIView):
    serializer_class = TrainChatbotSerializer
    def get(self, request):
        initialize_training_data()
        train_model()
        
        response = {"status": "success", "message": "Chatbot has been trained successfully."}
        serializer_class = TrainChatbotSerializer(response)
        
        return Response(serializer_class.data)

class PromptChatbot(generics.CreateAPIView):
    serializer_class = PromptChatbotSerializer
    
    def create(self, request, *args, **kwargs):
        model = load_trained_model()
        
        # Extract the prompt from the incoming request data
        message = request.data.get("prompt")
        if not message:
            return Response({"error": "No prompt provided"}, status=400)

        # Predict class and generate a response
        ints = predict_class(message, model)
        response = get_response(ints, data)

        # Prepare response data to return
        response_data = {
            "prompt": message,
            "response": response
        }

        # Serialize the response data using the serializer
        serializer = PromptChatbotSerializer(response_data)

        return Response(serializer.data, status=201)