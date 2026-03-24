"""
eğitilmiş modeli kullanarak yeni yorumların duygu analizini yapalım

"""
import numpy as np
import nltk # natural language tool kit
import matplotlib.pyplot as plt
from nltk.corpus import stopwords # gereksiz kelime listesi
from tensorflow.keras.models import load_model # eğitilmiş modeli yüklemek için
from tensorflow.keras.datasets import imdb # veri seti
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence # metni kelime dizisine cevirme

#model parametreleri
max_features = 10000 # en cok kullanilan 10 bin kelime
maxlen = 500 # yorum uzunlugu 500 kelimeye tamamlanacak

#stopwords kurtulma ve sozluklerin hazirlanmasi
nltk.download("stopwords") # nltk icinden ingilizce stopwords indriliyor
stop_words = set(stopwords.words("english")) # kucuk ve anlamsiz kelimeler ayiklanacak

# gerekli sozluklerin olusturulmasi: word to index ve index to word
word_index = imdb.get_word_index()
index_to_word = {index + 3: word for word, index in word_index.items()} # sayilardan kelimereler gecis
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"
word_to_index = {word: index for index, word in index_to_word.items()} # kelimelerden sayilara gecis

#egitilmis modeli yukleme
model = load_model("rnn_duygu_model.h5")
print("Model yuklendi.")

#tahmin yapacak fonksiyon
def predict_review_sentiment(review):
    """
    kullanicidan gelen metni temizle, modele uygun hale getir ve tahmin yap

    """
    #yorumu kucult ve kelime dizisine cevir
    words = text_to_word_sequence(review) # metni kelime dizisine cevir

    #stopwords temizleme
    cleaned = [
        word.lower() for word in words if word.isalpha() and word.lower() not in stop_words
    ]

    #her kelime egitilen sozlukten sayiya cevrilir, bilinmeyen kelimeler 2 ile gosterilir
    encoded = [word_to_index.get(word, 2) for word in cleaned]

    #yorum uzunlugu modele uygun hale getirilir
    padded = pad_sequences([encoded], maxlen = maxlen)

    #model ile tahmin yapilir
    prediction = model.predict(padded)[0][0]
    print(f"Tahmin: {'Pozitif' if prediction >= 0.5 else 'Negatif'} (Olasilik: {prediction:.4f})")


#kullanici girdisi ile tahmin yapalim
user_review = input("Bir film yorumu girin: ")
predict_review_sentiment(user_review)
