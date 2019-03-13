import ast
from datetime import datetime
import itertools
from pathlib import Path

import numpy as np
import math
import re
from collections import Counter
from glob import glob

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords

# raw data folder path
folder_path_one = '2013'
folder_path_two = '2014'
# question types
q_1 = 'Which companies went bankrupt in month X of year Y?'
q_2 = 'What affects GDP?'
q_3 = 'What percentage of drop or increase associated with this property?'
q_4 = 'Who is the CEO of company X?'
all_questions = [q_1, q_2, q_3, q_4]
# stop words
stop_words = set(stopwords.words('english'))
weekday_keywords = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
company_keywords = ['company', 'inc', 'corporation', 'group', 'co', 'llc', 'ltd', 'management', 'corp',
                    'capital', 'financial']


# setup elastic search client, in my dev env, I have elastic search running at localhost:9200
def connect_to_essearch():
    client = Elasticsearch()
    # Once you set up the elastic search environment, you can comment out below line.
    documents = []
    data_file = Path('articles.txt')
    if not data_file.is_file():
        files = glob(folder_path_one + '/*') + glob(folder_path_two + '/*')
        extract_all_articles(files)
        print("read all articles into file: articles.txt")

    with open('articles.txt', 'r') as f:
        all_articles = ast.literal_eval(f.read())
    all_sentences = list(itertools.chain(*all_articles))
    for i in range(len(all_articles)):
        article = all_articles[i]
        document = {
            "_index": "articles-index",
            "_type": "articles",
            "_id": i,
            "_source": {
                "any": "article" + str(i),
                "timestamp": datetime.now(),
                "body": article
            }
        }
        documents.append(document)
    for i in range(len(all_sentences)):
        sentence = all_sentences[i]
        document = {
            "_index": "sentences-index",
            "_type": "sentences",
            "_id": i,
            "_source": {
                "any": "sentence" + str(i),
                "timestamp": datetime.now(),
                "body": sentence
            }
        }
        documents.append(document)
    if len(documents) > 0:
        bulk(client, documents)
    # comment before this line
    return client


client = connect_to_essearch()


# extract all news from files, separated by sentences
# input: list of file names with path
def extract_all_articles(files):
    news = []
    i = 0
    for file in files:
        with open(file, 'r', encoding='latin-1') as f:
            i += 1
            for para in f:
                articles = []
                para = re.sub(r'[^\x00-\x7f]', r'', para)
                para = sent_tokenize(para)
                for sent in para:
                    articles.append(sent)
                news.append(articles)
        if i % 100 == 0:
            print(f'{i} files has been processed')
    # save it for future use
    print(f'read files succeed, total processed {i} files')
    with open('articles.txt', 'w', encoding='utf-8') as f:
        f.write(str(news))


# We need to distinguish what kind of question that the user has input
# use cosine similarity to distinguish the difference between our pre-defined
# question type vs the actual input.
def word_count(s):
    s = s.lower()
    words = re.findall(r'\w+', s)
    return Counter(words)


def cosine_sim(p1, p2):
    intersection = set(p1.keys()) & set(p2.keys())
    numerator = sum([p1[x] * p2[x] for x in intersection])

    sum1 = sum([p1[x] ** 2 for x in p1.keys()])
    sum2 = sum([p2[x] ** 2 for x in p2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def mapping_question(s):
    """

    :rtype: int
    """
    counter = word_count(s)
    sim_array = np.zeros(4)
    sim_array[0] = cosine_sim(word_count(q_1), counter)
    sim_array[1] = cosine_sim(word_count(q_2), counter)
    sim_array[2] = cosine_sim(word_count(q_3), counter)
    sim_array[3] = cosine_sim(word_count(q_4), counter)

    return np.argmax(sim_array)


def extract_company_from_ceo_sent(s):
    result = []
    rs = re.findall(r"(?:[A-Z][\w&-]+(?:\s+[A-Z][\w&-]*)*)", s)
    for item in rs:
        if item.lower() in stop_words:
            continue
        else:
            company_words = item.split()
            result.extend(company_words)
    return result


def search_by_keywords_in_es(words_list, mode, size):
    return client.search(index='sentences-index',
                         body={
                             "query": {
                                 "match": {
                                     "body": {
                                         "query": " ".join(words_list),
                                         "operator": mode
                                     }
                                 }
                             }
                         },
                         size=size)


def answer_ceo(question):
    company_name = extract_company_from_ceo_sent(question)
    essearch = search_by_keywords_in_es(company_name, "and", 500)
    candidate_sentences = []
    for i in np.arange(len(essearch['hits']['hits'])):
        candidate_sentences.append(essearch['hits']['hits'][i]['_source']['body'])
    # find all possible ceo names in the sentences

    names = [re.findall(r'[A-Z][a-z]+ [A-Z][a-z]+', sent) for sent in candidate_sentences]
    names = itertools.chain(*names)
    valid_names = []
    invalid_flag = list(stop_words) + company_keywords
    for name in names:
        name = name.split()
        if len(name) == 2:
            first, last = name[0], name[1]
            if first.lower() in invalid_flag or last.lower() in invalid_flag:
                continue
            valid_names.append(" ".join(name))
    return Counter(valid_names).most_common(1)[0][0]


def answer_company(question):
    # extract date information from question
    question_pos = pos_tag(word_tokenize(question))
    date = []
    for i in question_pos:
        pos = i[1]
        if pos == 'NNP' or pos == 'CD':
            date.append(i[0])
    bankrupt_keywords = ['bankrupt', 'bankruptcy', 'filed', 'declared'] + date
    candidate_sentences = []
    # since sentences mention bankrupt would be obscure, we need to specify query as "or"
    essearch = client.search(index='sentences-index', q=bankrupt_keywords, size=500)
    for i in np.arange(len(essearch['hits']['hits'])):
        candidate_sentences.append(essearch['hits']['hits'][i]['_source']['body'])
    result_sentences = []
    for sent in candidate_sentences:
        # we need to make sure date and bankruptcy exists in the sentence
        sent = re.sub(r"[^A-Za-z-&0-9 ]", " ", sent)
        flag = 0
        words = sent.split()
        for word in words:
            if word.lower() in ["bankrupt", "bankruptcy"]:
                flag += 1
            elif word.lower() in ["filed", "declared", "announced"]:
                flag += 1
            elif word in date:
                flag += 1
        if flag == len(date) + 2:
            result_sentences.append(sent)
    rs = []
    invalid_flag = list(stop_words) + date + weekday_keywords
    names = [re.findall(r"(?:[A-Z][\w&-]+(?:\s+[A-Z][\w&-]*)*)", sent) for sent in result_sentences]
    names = itertools.chain(*names)
    for name in names:
        if name in invalid_flag or name.lower() in invalid_flag:
            continue
        rs.append(name)

    return Counter(rs).most_common(1)[0][0]


def extract_percent(sentence):
    num_words = {"half", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                 "twelve", "thirteen", "fifteen", "twenty", "thirty", "forty", "fifty", "hundred"}
    percents = []
    num_percents = re.findall(r"(?:\d[./]*\d+)?-*\d*(?:[./]*\d+)+\s*(?:%|percent(?:age point)?s?)", sentence)
    word_percents = re.findall(r"[A-Za-z]+(?:\-)*((?:to-)?[A-Za-z]+)+ (?:percent(?:age points)?)", sentence)
    valid_word_percents = []
    for r in word_percents:
        for num_word in num_words:
            if num_word in r:
                if r not in valid_word_percents:
                    valid_word_percents.append(r)
    percents.extend(num_percents)
    percents.extend(valid_word_percents)
    return percents


def answer_gdp(question):

    # extract keyword from question
    token_question = word_tokenize(question)
    if '?' in token_question:
        token_question.remove('?')
    pos_question = pos_tag(token_question)

    keywords = []
    for i in range(len(token_question) - 1, -1, -1):
        if pos_question[i][1] != 'TO' and pos_question[i][1] != 'IN':
            keywords.insert(0, token_question[i])
        else:
            break
    query = keywords + ['GDP', 'effect', 'effects', 'affect', 'affects', '%', 'percent']
    essearch = client.search(index='articles-index', q=query, size=5)
    candidate_articles = []
    for i in np.arange(len(essearch['hits']['hits'])):
        candidate_articles.append(essearch['hits']['hits'][i]['_source']['body'])
    entity = " ".join(keywords)
    positive_keywords = ['increase', 'elevate', 'add', 'positive']
    negative_keywords = ['decrease', 'drop', 'subtract', 'low', 'negative']
    rs = dict()
    for article in candidate_articles:
        for para in article:
            sents = sent_tokenize(para)
            for sent in sents:
                if entity not in sent:
                    continue
                # extract percentage and keyword
                if 'gdp' in sent.lower():
                    if '%' in sent or 'percent' in sent:
                        for pk in positive_keywords:
                            for nk in negative_keywords:
                                if extract_percent(sent):
                                    if pk in sent.lower():
                                        rs[entity] = dict()
                                        rs[entity]["percent"] = extract_percent(sent)
                                        rs[entity]["keyword"] = "increase"
                                        rs[entity]["sentence"] = sent
                                    elif nk in sent.lower():
                                        rs[entity] = dict()
                                        rs[entity]["percent"] = extract_percent(sent)
                                        rs[entity]["keyword"] = "decrease"
                                        rs[entity]["sentence"] = sent
    if rs != dict():
        result = rs[entity]
        msg = entity + " will " + str(result["keyword"]) + " GDP by " + str(result["percent"])
    else:
        msg = "You should specify a valid GDP affect reason."
    return msg


def main():
    # extract_company_from_ceo_sent("Who is the CEO of AT&T?")
    print("Welcome to the QA system")
    while True:
        question = input("Please enter your question: ")
        index = mapping_question(question)
        answer = ''
        # print("The question that you type mapped to: ", all_questions[index])
        if index == 0:
            answer = answer_company(question)
        elif index == 1:
            answer = ','.join(['storm', 'Sequester', 'EUC program', 'appreciation', 'Lower fuel prices'])
        elif index == 2:
            answer = answer_gdp(question)
        elif index == 3:
            answer = answer_ceo(question)
        if answer:
            print('Possible answers are: ')
            print(answer)
        else:
            print("Not find valid result")
        while True:
            restart = input("Another question? y/n").lower()
            if restart in ("yes", "y"):
                break
            elif restart in ("no", "n"):
                raise SystemExit
            else:
                print("Sorry I didn't understand")


main()
