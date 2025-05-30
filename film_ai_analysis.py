import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# CSV dosyasını oku
df = pd.read_csv("yorumlar.csv")

# Eğer 'Comment' adında bir sütun varsa onu 'text' olarak değiştir
df = df.rename(columns={"Comment": "text"})

# Sentiment analiz fonksiyonu
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

# Sentiment skorlarını hesapla
df['sentiment'] = df['text'].apply(get_sentiment)

# Ortalama skoru yazdır
avg_sentiment = df['sentiment'].mean()
print("Ortalama Duygu Skoru:", avg_sentiment)

# Histogram çiz
plt.hist(df['sentiment'], bins=20, color='skyblue')
plt.title("Yorumlardaki Duygu Dağılımı")
plt.xlabel("Skor")
plt.ylabel("Yorum Sayısı")
plt.show()
