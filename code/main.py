import csv
from collections import Counter

import pandas as pd
import emoji
import random
import string

from nltk.stem.isri import ISRIStemmer
import nltk

from nltk.corpus import stopwords
import re
import unicodedata
from googletrans import Translator
import tweepy
from pyarabic.araby import strip_tashkeel, normalize_ligature
import ast
from mtranslate import translate

# df = pd.read_csv('dataset/preprocessed_tweets.csv')
# print(df_copy.head())
# print(df_copy.info())

df = pd.read_csv('dataset/Emotional-Tone-Dataset.csv')
df_copy = df.copy()

# Remove null values
df_copy.dropna(subset=[' TWEET'], inplace=True)
# fff = df_copy.iloc[9800:]


def clean_arabic_sentence(sentence):
    punctuations = string.punctuation + '،؛؟'

    # white space removal
    sentence = re.sub(r'\s+', ' ', sentence).strip()

    # remove punctuation
    removed_punc = ''.join(char for char in sentence if char not in punctuations)

    # remove tashkeel and elongation
    removed_tashkeel = strip_tashkeel(removed_punc)
    normalized_ligature = normalize_ligature(removed_tashkeel)
    removed_elongation = re.sub(r'(.)\1+', r'\1', normalized_ligature)

    # Tokenize the sentence into words
    words = nltk.word_tokenize(removed_elongation)


    stop_words_egyptian = ['انا', 'انت', 'انتي', 'انه', 'انها', 'لأن', 'ازاي', 'ازي', 'لو', 'مش',
                           'كدة', 'كده', 'كلنا', 'كله', 'كلهم', 'كلهما', 'كلهن', 'كلي',
                           'كلو', 'ليه',
                           'لوحده', 'ليك', 'ليهم', 'ليهما', 'معاه', 'معاها', 'معاهم', 'معاهما', 'معانا', 'معاك',
                           'معاهو',
                           'معاهمو', 'معاهن', 'نحنا', 'هما', 'هماك', 'هماه', 'همايا', 'همم', 'همه', 'همو', 'همها', 'هي',
                           'هو',
                           'هيا', 'هيك', 'هيكم', 'هيكوا', 'هيهات', 'وانا', 'وانت', 'وانتي', 'وانه', 'وانها', 'ولأن',
                           'وازاي',
                           'وازي', 'ولو', 'ومش' 'وكدة',
                           'وكده',
                           'وكلنا', 'وكله', 'وكلهم', 'وكلهما', 'وكلهن', 'وكلي', 'وكلو', 'وليه', 'ولوحده', 'وليك',
                           'وليهم',
                           'وليهما', 'ومعاه', 'ومعاها', 'ومعاهم', 'ومعاهما', 'ومعانا', 'ومعاك', 'ومعاهو', 'ومعاهمو',
                           'ومعاهن',
                           'ونحنا', 'وهما', 'وهماك', 'وهماه', 'وهمايا', 'وهمم', 'وهمه', 'وهمو', 'وهمها', 'وهي', 'وهو',
                           'وهيا',
                           'وهيك', 'وهيكم', 'وهيكوا', 'وهيهات'] + ["ان"] + ["علي"] + ["الي"] + ["اله"] + ["انا"] + [
                              "مش"] + ["دي"] + ["انت"] + ["ايه"] + ["وله"] + \
                          ["ده"] + ["انه"] + ["الا"] + ["اني"] + ["كده"] + ["اي"] + ["او"] + ["واله"] + ["عشان"] + [
                              "معكم"] + ["يعني"] + ["حتي"] + ["كنت"] + ["احنا"] + ["دا"] + ["ليه"] + ["وانا"] \
                          + ["انك"] + ["اذا"] + ["زي"] + ["ازاي"] + ["يكون"] + ["منك"] + ["كانت"] + ["ال"] + [
                              "كلها"] + ["كلنا"] + ["كله"] + ["منك"] + ["اللي"] + \
                          ["مصر"] + ["الله"] + ["السعوديه"] + ["قطر"] + ["ايران"] + ["البرازيل"] + ["اللهم"] + [
                              "حد"] + ["حاجه"] + ["d"] + ["والله"]
    # ["الاوليمبياد"]
    # Remove stop words
    stop_words = stopwords.words('arabic') + stop_words_egyptian
    removed_stop_words = [word for word in words if word not in stop_words]
    arabic_words = []

    # Remove english words
    for token in removed_stop_words:
        if not re.search(r'[a-zA-Z0-9]', token):
            arabic_words.append(token)

    return arabic_words


def random_deletion(sentence):
    words = sentence.split()
    if len(words) == 1:
        return sentence
    delete_index = random.randint(0, len(words) - 1)
    del words[delete_index]
    augmented_sentence = ' '.join(words)
    return augmented_sentence


def compare_column_values(df1, df2, column_name1, column_name2):
    result = []
    if len(df1) == len(df2):
        for i in range(len(df1)):
            if df1.iloc[i][column_name1] == df2.iloc[i][column_name2]:
                result.append(True)
            else:
                result.append(False)
    else:
        result.append(False)
    return result


def remove_emojis(input_string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r"", input_string)


def get_tweets():
    consumer_key = 'eztIAXCyqwELFRZhWhUyRBkw1'
    consumer_secret = 'vWRigzXKNdXaMpTUMlrwZouMt2FWQC6mEADc8HOgi7LN8Hhd95'
    access_token = 'yNpkO50kGe9pFcYL9L9Lm21g9LMt40'
    access_token_secret = '9x3LKpLtA9LC20CCU8e2sQCT3JRTRDuaqH9xSp6wPQoVy'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    keyword = 'كورونا'
    language = 'ar'
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang=language).items()

    # tweets_df = pd.DataFrame(columns=[' TWEET', 'Username', 'Timestamp'])
    for tweet in tweets:
        print(tweet.text)
        tweet_text = tweet.text
        username = tweet.user.screen_name
        timestamp = tweet.created_at

    # tweets_df = tweets_df.append({' TWEET': tweet_text, 'Username': username, 'Timestamp': timestamp},
    # ignore_index=True)

    # tweets_df.to_csv('tweets.csv', index=False, encoding='utf-8-sig')


def eliminate_speech_effect(text):
    text = strip_tashkeel(text)
    text = normalize_ligature(text)
    text = re.sub(r'(.)\1+', r'\1', text)

    return text


def findMostFrequentWords():
    df[' TWEET'] = df[' TWEET'].apply(ast.literal_eval)
    all_words = [word for tokens in df[' TWEET'] for word in tokens]
    word_freq = Counter(all_words)
    word_freq_df = pd.DataFrame.from_dict(word_freq, orient='index', columns=['Frequency'])
    word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)

    print(word_freq)

    word_freq_list = list(word_freq.items())
    csv_file = "word_freq.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        for word, freq in word_freq_list:
            writer.writerow([freq, word])

    word_freq.to_csv('dataset/word_frequencies.csv', index=False)


def findMostFreqWordPerLabel():
    label_groups = df.groupby(' LABEL')

    word_freq_by_label = {}

    for label, group in label_groups:
        corpus = ' '.join(group[' TWEET'].astype(str).tolist())  # Convert values to strings before joining

        tokens = nltk.word_tokenize(corpus)  # Tokenize the text
        tokens = [word.lower() for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens

        stop_words_egyptian = ['انا', 'انت', 'انتي', 'انه', 'انها', 'لأن', 'ازاي', 'ازي', 'لو', 'مش',
                               'كدة', 'كده', 'كلنا', 'كله', 'كلهم', 'كلهما', 'كلهن', 'كلي',
                               'كلو', 'ليه',
                               'لوحده', 'ليك', 'ليهم', 'ليهما', 'معاه', 'معاها', 'معاهم', 'معاهما', 'معانا', 'معاك',
                               'معاهو',
                               'معاهمو', 'معاهن', 'نحنا', 'هما', 'هماك', 'هماه', 'همايا', 'همم', 'همه', 'همو', 'همها',
                               'هي',
                               'هو',
                               'هيا', 'هيك', 'هيكم', 'هيكوا', 'هيهات', 'وانا', 'وانت', 'وانتي', 'وانه', 'وانها', 'ولأن',
                               'وازاي',
                               'وازي', 'ولو', 'ومش' 'وكدة',
                               'وكده',
                               'وكلنا', 'وكله', 'وكلهم', 'وكلهما', 'وكلهن', 'وكلي', 'وكلو', 'وليه', 'ولوحده', 'وليك',
                               'وليهم',
                               'وليهما', 'ومعاه', 'ومعاها', 'ومعاهم', 'ومعاهما', 'ومعانا', 'ومعاك', 'ومعاهو', 'ومعاهمو',
                               'ومعاهن',
                               'ونحنا', 'وهما', 'وهماك', 'وهماه', 'وهمايا', 'وهمم', 'وهمه', 'وهمو', 'وهمها', 'وهي',
                               'وهو',
                               'وهيا',
                               'وهيك', 'وهيكم', 'وهيكوا', 'وهيهات'] + ["ان"] + ["علي"] + ["الي"] + ["اله"] + ["انا"] + [
                                  "مش"] + ["دي"] + ["انت"] + ["ايه"] + ["وله"] + \
                              ["ده"] + ["انه"] + ["الا"] + ["اني"] + ["كده"] + ["اي"] + ["او"] + ["واله"] + ["عشان"] + [
                                  "معكم"] + ["يعني"] + ["حتي"] + ["كنت"] + ["احنا"] + ["دا"] + ["ليه"] + ["وانا"] \
                              + ["انك"] + ["اذا"] + ["زي"] + ["ازاي"] + ["يكون"] + ["منك"] + ["كانت"] + ["ال"] + [
                                  "كلها"] + ["كلنا"] + ["كله"] + ["منك"] + ["اللي"] + \
                              ["مصر"] + ["الله"] + ["السعوديه"] + ["قطر"] + ["ايران"] + ["البرازيل"] + ["اللهم"] + [
                                  "حد"] + ["حاجه"] + ["d"] + ["والله"]

        stop_words = stopwords.words('arabic') + stop_words_egyptian
        words = [word for word in tokens if word not in stop_words]

        freq_dist = nltk.FreqDist(words)
        word_freq_by_label[label] = freq_dist

    most_frequent_words_by_label = {}

    for label, freq_dist in word_freq_by_label.items():
        most_common_words = freq_dist.most_common(1)  # Get the most frequent word
        most_frequent_words_by_label[label] = most_common_words[0][0]

    print(most_frequent_words_by_label)


def convert_emojis_to_text(input_string):
    english_text = emoji.demojize(input_string, delimiters=(" ", " "))
    clean_english_texts = english_text.replace("_", " ").replace(":", "").strip()
    arabic_text = translate(clean_english_texts, 'ar', 'en')
    return arabic_text


df_copy[' TWEET'] = df_copy[' TWEET'].apply(remove_emojis)
# fff[' TWEET'] = fff[' TWEET'].apply(convert_emojis_to_text)
df_copy[' TWEET'] = df_copy[' TWEET'].apply(clean_arabic_sentence)
# same_values = compare_column_values(df_augmented, df_copy, ' TWEET', ' TWEET')

# print(df_copy[" TWEET"][539])
# print(df_augmented[" TWEET"][539])

df_copy.to_csv('dataset/preprocessed_tweets.csv', index=False)

# testing = "يا ديني ع الضييييييييييحك  هههههههههههههههههههههههههههههههههههههههههههههههههههههههههههههههههههه"
# testing_result = eliminate_speech_effect(testing)
# print(testing_result)


# text = ":  علي فكره تريد  الموت 😨 ارمي  نفسك ب  البحر ستري انك  تصارع لكي  تعيش 🤕 انت  لا تريد ان تموت ✋بل تريد  قتل ماتركه  الناس في…"
# text_string = convert_emojis_to_text(text)
# print(text_string)

