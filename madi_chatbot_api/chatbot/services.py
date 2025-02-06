import json, os
import numpy as np
import nltk
import random
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


data = []
words = []
classes = []
documents = []
ignore_words = ["?", "!"]
lemmatizer = WordNetLemmatizer()

train_x = []
train_y = []

def load_data():
    with open('data.json', 'r') as file:
        data = json.load(file)
        
    return data

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
    
def load_trained_model():
    # if os.path.exists("chatbot_model.h5"):
    #     return load_model("chatbot_model.h5")
    if os.path.exists("chatbot_model.keras"):
        return load_model("chatbot_model.keras")
    else:
        raise FileNotFoundError("Model file 'chatbot_model.h5' not found. Please train the chatbot first.")

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
    
def setup_preprocessing_of_data():
    global data
    data = load_data()
    # Step 2: Preprocessing
    def preprocess_data():
        global words, classes, documents
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
        preprocess_data()
        
        # Save words and classes
        pickle.dump(words, open("words.pkl", "wb"))
        pickle.dump(classes, open("classes.pkl", "wb"))
        
    save_preprocessed_data()

def setup_training_data():
    global train_x, train_y, documents, lemmatizer
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
    
def initialize_preprocessed_data():
    setup_nltk()
    setup_preprocessing_of_data()

def initialize_training_data():
    setup_training_data()

def build_model():
    model = Sequential()
    model.add(Dense(256, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation="softmax"))

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    
    return model

def train_model():
    model = build_model()
    model.fit(train_x, train_y, epochs=1024, batch_size=32, verbose=1)
    # model.save("chatbot_model.h5")
    model.save("chatbot_model.keras")
    
def start_training_process():
    initialize_preprocessed_data()
    initialize_training_data()
    train_model()
    
