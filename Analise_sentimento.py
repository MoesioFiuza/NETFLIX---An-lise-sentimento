import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import pycountry

caminho_arquivo = r'cole aqui o caminho para o arquivo csv'
df = pd.read_csv(caminho_arquivo)

def traduzir_pais(nome_pais):
    if pd.isna(nome_pais):
        return nome_pais
    try:
        pais = pycountry.countries.get(alpha_2=nome_pais)
        if pais:
            return pais.name
        else:
            return nome_pais
    except KeyError:
        return nome_pais

df['country'] = df['country'].apply(lambda x: ', '.join([traduzir_pais(pais.strip()) for pais in x.split(',')]) if pd.notna(x) else x)

df['sentiment'] = df['describle'].dropna().apply(lambda x: TextBlob(x).sentiment.polarity)

all_descriptions = ' '.join(df['describle'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_descriptions)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud das Descrições')

plt.figure(figsize=(10, 5))
sns.histplot(df['sentiment'], bins=30, kde=True, color='blue')
plt.title('Distribuição dos Sentimentos das Descrições')
plt.xlabel('Polaridade do Sentimento')
plt.ylabel('Frequência')


plt.figure(figsize=(10, 5))
sns.boxplot(x='type', y='sentiment', data=df, palette=['blue', 'orange'])
plt.title('Distribuição do Sentimento por Tipo')
plt.xlabel('Tipo')
plt.ylabel('Polaridade do Sentimento')

plt.figure(figsize=(10, 5))
sns.scatterplot(x='rating', y='sentiment', data=df, color='orange')
plt.title('Sentimento vs. Rating')
plt.xlabel('Rating')
plt.ylabel('Polaridade do Sentimento')


plt.figure(figsize=(10, 5))
sns.boxplot(x='country', y='sentiment', data=df, palette='coolwarm')
plt.title('Distribuição do Sentimento por País')
plt.xlabel('País')
plt.ylabel('Polaridade do Sentimento')
plt.xticks(rotation=90)
plt.show()

correlacoes = df[['sentiment', 'rating']].corr()
print('Correlação entre Sentimento e Rating:')
print(correlacoes)
