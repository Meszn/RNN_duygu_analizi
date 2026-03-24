"""
RNN ile Duygu Analizi
Problem Tanimi: Bir metnin duygu durumunu (pozitif veya negatif) tahmin etmek.
    IMDB veri setini kullanarak bir RNN modeli egitilecek.
    Veri seti: IMDB Movie Reviews
    -this movie is awesome -> pozitif
    -this movie is terrible -> negatif

RNN: Tekrarlayan Sinir Agi (Recurrent Neural Network)
    - Sirali veriler üzerinde calisir, metin gibi bilgilerde önceki bilgileri hatirlayarak sonraki tahminleri yapmaya calisir.

Veri Seti: IMDB veri seti film yorumlarindan olusur.
    - 50.000 pozitif ve negatif film yorumu icerir.
    - 25.000 pozitif film yorumu
    - 25.000 negatif film yorumu
    - Her yorum 1 ile 500 arasinda kelime icerir.
    - Her kelime 1 ile 10.000 arasinda bir tam sayi ile temsil edilir.
    - 0: bosluk, 1: <start>, 2: <stop>, 3: <unknown>

"""
import numpy as np
import nltk # natural language tool kit
import matplotlib.pyplot as plt
from nltk.corpus import stopwords # gereksiz kelime listesi
from tensorflow.keras.models import Sequential # base model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense 
from tensorflow.keras.datasets import imdb # veri seti
from tensorflow.keras.preprocessing.sequence import pad_sequences

# stopwords (gereksiz kelimeler) listesi belirle
nltk.download("stopwords") # nltk icinden ingilizce stopwords indriliyor
stop_words = set(stopwords.words("english")) # kucuk ve anlamsiz kelimeler ayiklanacak

# model parametreleri
max_features = 10000 # en cok kullanilan 10 bin kelime
maxlen = 500

# load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = max_features) # train/test ayrilmis sekilde veri gelir

# ornek veri incelemesi
original_word_index = imdb.get_word_index()

# sayi kelime donusum sozlugu hazirlama
inv_word_index = { index + 3: word for word, index in original_word_index.items()}
inv_word_index[0] = "<PAD>" # 0: bosluk padding
inv_word_index[1] = "<START>" # 1: cumle baslangici
inv_word_index[2] = "<UNK>" # 2: bilinmeyen kelime
# inv_word_index[3] -> great: 65

# sayi dizisini kelimelere ceviren fonksiyon
def decode_review(encoded_review):
    return " ".join([inv_word_index.get(i, "?") for i in encoded_review])

movie_index = 0
# ilk egitim verisini yazdiralim
print("ilk yorum: (sayi dizisi)")
print(X_train[movie_index])

print("ilk yorum: (kelimelerle)")
print(decode_review(X_train[movie_index]))

print(f"Label: {'Pozitif' if y_train[movie_index]== 1 else 'Negatif'}")

# gerekli sozluklerin olusturulmasi: word to index ve index to word
word_index = imdb.get_word_index()
index_to_word = {index + 3: word for word, index in word_index.items()} # sayilardan kelimereler gecis
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"
word_to_index = {word: index for index, word in index_to_word.items()} # kelimelerden sayilara gecis

# data preprocessing (veri on isleme)
def preprocess_review(encoded_review):
    # sayilari kelimelere cevir
    words = [index_to_word.get(i, "") for i in encoded_review if i >= 3]

    # sadece harflerden olusan ve stop words olmayanlari al
    cleaned = [
        word.lower() for word in words if word.isalpha() and word.lower() not in stop_words
    ]

    # tekrardan temizlenmis metni sayilara cevir
    return [word_to_index.get(word, 2) for word in cleaned]

# veriyi temizle ve sabit uzunlugu pad et
X_train = [preprocess_review(review) for review in X_train]
X_test = [preprocess_review(review) for review in X_test]

# pad sequence
"""
merhaba bugun hava cok guzel
merhaba, naber, 0, 0, 0
"""
X_train = pad_sequences(X_train, maxlen = maxlen)
X_test = pad_sequences(X_test, maxlen= maxlen)

# RNN Modeli olusturma
model = Sequential() # base model: katmanlari sirali olarak eklemek icin

# embedding katmani: kelime indexlerini 32 boyutlu bir vektore donusturur
model.add(Embedding(input_dim = max_features, output_dim = 32, input_length= maxlen))

# simplernn katmani: metni sirayla isler ve baglam iliskisini ogrenir
model.add(SimpleRNN(units = 32)) # cell (noron) sayisi 

# output katmani: binary classification: sigmoid, 1 noron
"""
negatif -> 0.7
"""
model.add(Dense(1, activation = "sigmoid"))

# model compile
model.compile(
    optimizer = "adam", # agirlik guncellemesi icin kullanilan algoritma
    loss = "binary_crossentropy", # kayip fonksiyonu
    metrics = ["accuracy"] # degerlendirme metrigi
)

print(model.summary())

# training
history = model.fit(
    X_train, y_train, # girdi ve cikti veri
    epochs = 2, # egitim tekrar sayisi yani tum veriyi 2 kere egit
    batch_size = 64, # torba, ayni anda islenecek ornek sayisi yani 64 lu paketler halinde isle
    validation_split = 0.2 # %20 yi dogrulama icin ayir
)

#model evaluation
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # kayip grafi
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Egitim Kaybi")
    plt.plot(history.history["val_loss"], label="Dogrulama Kaybi")
    plt.title("Egitim ve Dogrulama Kaybi")
    plt.xlabel("Epoch")
    plt.ylabel("Kayip")
    plt.legend()

    # dogruluk grafi
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Egitim Dogrulugu")
    plt.plot(history.history["val_accuracy"], label="Dogrulama Dogrulugu")
    plt.title("Egitim ve Dogrulama Dogrulugu")
    plt.xlabel("Epoch")
    plt.ylabel("Dogruluk")
    plt.legend()

    plt.show()

plot_history(history)

#test verisi ile model degerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Kaybi: {test_loss:.4f}, Test Dogrulugu: {test_acc:.4f}")

#eğitilen modelin kaydedilmesi
model.save("rnn_duygu_model.h5")
print("Model kaydedildi: rnn_duygu_model.h5")

