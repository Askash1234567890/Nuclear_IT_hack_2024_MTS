import numpy as np
from aiogram import types
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext
from loader import dp, bot

from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
from tqdm import tqdm
from wordcloud import WordCloud
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import pandas as pd
import re
import pymorphy2
import nltk
import json
from sentence_transformers import SentenceTransformer

###############################################################################################
##############################____BASE_PREPROCESSING____#######################################
nltk.download('stopwords')
stopwords_ru = stopwords.words("russian")
reg_stop = '|'.join(map(lambda x: f'(^{x}$)', stopwords_ru)) + '|(^из-за$)|(^из-под$)|(^.*ий$)' 
reg = r'[^А-Яа-я /-]'
bad_reg = r'(.*ху.*)|(.*пиз.*)|(.*еба.*)|(.*еби.*)' + rf'|{reg_stop}'
###############################################################################################
class FormStates(StatesGroup):
    waiting_for_file = State()

class GradSearch(object):
    def __init__(self, a=1.0, b=0.0, x0=1.0, l=1e-2):
        self.a = a
        self.b = b
        self.x0 = x0
        self.l = l
        self.loss = []

    def f(self, x, y, a, b, x0):
        return 0.5 * pow(y - a * np.exp(-x / x0) - b, 2).sum()

    def dfda(self, x, y, a, b, x0):
        return ((a * np.exp(-x / x0) + b - y) * np.exp(-x / x0)).sum()

    def dfdb(self, x, y, a, b, x0):
        return (a * np.exp(-x / x0) + b - y).sum()

    def dfdx0(self, x, y, a, b, x0):
        return ((a * np.exp(-x / x0) + b - y) * a * np.exp(-x / x0) * x / pow(x0, 2)).sum()

    def fit(self, x, y, epochs=1_000):
        self.loss.append(self.f(x, y, self.a, self.b, self.x0))
        for epoch in tqdm(range(epochs)):
            self.a -= self.l * self.dfda(x, y, self.a, self.b, self.x0)
            self.b -= self.l * self.dfdb(x, y, self.a, self.b, self.x0)
            self.x0 -= self.l* self.dfdx0(x, y, self.a, self.b, self.x0)
            self.loss.append(self.f(x, y, self.a, self.b, self.x0))

    def get_a_b_x0(self):
        return self.a, self.b, self.x0
    
def prepare_model(words, name):
    

    words = [re.sub(reg, '', word.lower()).strip() for word in words]
    morph = pymorphy2.MorphAnalyzer()
    dict_lemmas_to_words = {}
    lemmas = []
    for word in words:
        data = word.split()
        if len(data) == 1:
            lemmas.append(morph.parse(word)[0].normal_form)
        else:
            lemmas.append(' '.join([morph.parse(w)[0].normal_form for w in data if not(re.search(bad_reg, w))]))
            if word:
                dict_lemmas_to_words[lemmas[-1]] = word

    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = lemmas
    word_vectors = model.encode(sentences)

    ran = range(1, min(len(lemmas) + 1, 100))
    inertia_df = pd.DataFrame(data=[], index=ran, columns=['inertia'])
    effective_clusters = -1
    for n_clusters in tqdm(ran):
        try:
            effective_clusters += 1
            centers, indices = kmeans_plusplus(np.array(word_vectors), n_clusters=n_clusters, random_state=10)
            kmeans = KMeans(n_clusters=n_clusters,  random_state=42)
            cluster_labels = kmeans.fit_predict(word_vectors)
            inertia_df.loc[n_clusters] = kmeans.inertia_
        except:
            break
    inertia_df = inertia_df.iloc[:effective_clusters]
    inertia_arr = np.array(inertia_df).flatten()
    inertia_derivative = inertia_arr[:-1] - inertia_arr[1:]
    x, y = np.array(range(inertia_derivative.shape[0])), np.array(inertia_derivative).flatten()
    y_norm = y.max()
    y /= y_norm

    grad = GradSearch(a=1.0, b=0.0, x0=1.0, l=1e-2)
    grad.fit(x, y, epochs=100_000)
    a, b, x0 = grad.get_a_b_x0()

    ind = None
    porog = (a + b) / np.e
    for i, arg in enumerate(x):
        if a * np.exp(- arg / x0) + b < porog:
            ind = i
            break

    n_clusters = ind + 1
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(word_vectors)

    df = pd.DataFrame(lemmas, columns=['word'])
    df['cluster'] = cluster_labels

    counter = {}
    for i in range(n_clusters):
        w = np.random.choice(df[df['cluster'] == i]['word'])
        amount = df[df['cluster'] == i].shape[0]
        counter[w] = amount

    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='magma').generate_from_frequencies(counter)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(name)
    plt.savefig('cloud.png', format='png')
    plt.close()
    with open('file.json', 'w') as json_file:
        json.dump(counter, json_file, indent=4)

@dp.message_handler(commands=['start', 'help'])
async def hi(message: types.Message):
    markup = types.ReplyKeyboardRemove()
    await message.answer('Приветствую! Меня зовут Керил, я чат-бот предназначенный для создания облака слов на основе пользовательских ответов на вопросы'
                         '\n\nОтправь мне список слов в формате csv файла, чтобы я мог создать облако', reply_markup=markup)

@dp.message_handler(content_types=types.ContentType.DOCUMENT, state='*')
async def handle_document(message: types.Message, state: FSMContext):
    if message.document.mime_type == 'text/csv':
        await state.set_state(FormStates.waiting_for_file)
        await message.document.download(destination_file='received_file.csv')
        await message.reply("Данные обрабатываются. Подождите...")
        try:
            words = pd.read_csv('received_file.csv')
            res_words = []
            for column in words.columns:
                res_words.append(words[column].fillna("").tolist())
        except Exception as e:
            await message.reply(f"Ошибка при чтении файла: {str(e)}")
            await state.finish()
        for i in range(1, len(res_words)):
            try:
                prepare_model(res_words[i], words.columns[i])
                # Send the plot back to user
                with open('cloud.png', 'rb') as plot_file:
                    await message.reply_photo(photo=plot_file)
                with open("file.json", "r") as file:
                # Отправка документа пользователю
                    await bot.send_document(chat_id=message.chat.id, document=file)
            except Exception as e:
                if str(e) == 'zero-size array to reduction operation maximum which has no identity':
                    await message.reply(f"В вопросе {words.columns[i]} нет слов")
                await message.reply(f"Ошибка при чтении файла: {str(e)}")
        await state.finish()
    elif(message.document.mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'):
        await state.set_state(FormStates.waiting_for_file)
        await message.document.download(destination_file='received_file.xlsx')
        await message.reply("Данные обрабатываются. Подождите...")
        try:
            words = pd.read_excel('received_file.xlsx')
            res_words = []
            for column in words.columns:
                res_words.append(words[column].fillna("").tolist())
        except Exception as e:
            await message.reply(f"Ошибка при чтении файла: {str(e)}")
            await state.finish()
        for i in range(1, len(res_words)):
            try:
                prepare_model(res_words[i], words.columns[i])
                # Send the plot back to user
                with open('cloud.png', 'rb') as plot_file:
                    await message.reply_photo(photo=plot_file)
                with open("file.json", "r") as file:
                # Отправка документа пользователю
                    await bot.send_document(chat_id=message.chat.id, document=file)
            except Exception as e:
                if str(e) == 'zero-size array to reduction operation maximum which has no identity':
                    await message.reply(f"В вопросе {words.columns[i]} нет слов")
                await message.reply(f"Ошибка при чтении файла: {str(e)}")
        await state.finish()
    elif(message.document.mime_type == 'application/json'):
        await state.set_state(FormStates.waiting_for_file)
        await message.document.download(destination_file='received_file.json')
        await message.reply("Данные обрабатываются. Подождите...")
        try:
            words = pd.read_json('received_file.json')
            res_words = []
            for column in words.columns:
                res_words.append(words[column].fillna("").tolist())
        except Exception as e:
            await message.reply(f"Ошибка при чтении файла: {str(e)}")
            await state.finish()
        for i in range(len(res_words)):
            try:
                prepare_model(res_words[i], words.columns[i])
                # Send the plot back to user
                with open('cloud.png', 'rb') as plot_file:
                    await message.reply_photo(photo=plot_file)
                with open("file.json", "r") as file:
                # Отправка документа пользователю
                    await bot.send_document(chat_id=message.chat.id, document=file)
            except Exception as e:
                if str(e) == 'zero-size array to reduction operation maximum which has no identity':
                    await message.reply(f"В вопросе {words.columns[i]} нет слов")
                await message.reply(f"Ошибка при чтении файла: {str(e)}")
        await state.finish()
    else:
        await message.reply("Пожалуйста, пришлите файл формата .csv/.json/.xlsx")



