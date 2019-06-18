from abc import ABC, abstractmethod

from gensim.test.utils import datapath, get_tmpfile
import gensim
from gensim.corpora import WikiCorpus, MmCorpus
from gensim.corpora.wikicorpus import extract_pages, filter_wiki
from gensim.corpora.textcorpus import TextCorpus
from gensim.models import FastText
from gensim.scripts import make_wiki
from gensim.test.utils import datapath
from gensim.corpora.dictionary import Dictionary
import mmap
from io import UnsupportedOperation
from zipfile import BadZipfile
import gensim
import nltk.data
from nltk.corpus import stopwords
import argparse
import os
import re
import logging
import sys
import multiprocessing as mp
import difflib as diff
from gensim import corpora
from gensim.models import Word2Vec
from gensim.models import FastText
# uncomment if gensim is installed
# !pip install gensim
import gensim
# Need the interactive Tools for Matplotlib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


class Cleaner(meta=ABC.Meta):

    def __init__(self, search_patterns, replacement):
        self._search_patterns = search_patterns
        if replacement is not None:
            self.with_replacement = replacement
        else:
            self.with_replacement = ' '

    @abstractmethod
    def _remove(self, sentence, search_pattern, **kwargs):
        pass


class Punctuation(Cleaner):

    def __init__(self, target_punctuation, replacement):
        super().__init__(search_patterns=target_punctuation, replacement=replacement)

    def _remove(self, sentence, search_pattern, **kwargs):
        sentence = sentence.replace(search_pattern, self.with_replacement)

        return sentence

    def remove_punctuation_from(self, sentence):
        """
                Delete punctuation from given text corpus
        """
        for punct_ in self._search_patterns:
            sentence = self._remove(sentence=sentence, search_pattern=punct_)

        return sentence


class Umlaut(Cleaner):

    def __init__(self, search_patterns, replacement):
        super().__init__(search_patterns, replacement)

    def _remove(self, sentence, search_pattern, **kwargs):
        sentence = sentence.replace(search_pattern, self.with_replacement)

        return sentence

    def remove_umlauts_from(self, sentence):
        for umlaut in self._search_patterns:
            sentence = self._remove(sentence, search_pattern=umlaut)

        return sentence


class Stopwords(Cleaner):
    def __init__(self, search_patterns, replacement):
        super().__init__(search_patterns, replacement)

    def _remove(self, sentence, search_pattern, **kwargs):
        sentence = sentence.replace(search_pattern, self.with_replacement)

        return sentence

    def remove_stopwords_from(self, sentence):
        for stopword in self._search_patterns:
            sentence = self._remove(sentence, stopword)

        return sentence


class JunkText(Cleaner):
    def __init__(self, search_patterns, replacement):
        super().__init__(search_patterns, replacement)

    def _remove(self, sentence, search_pattern, **kwargs):
        sentence = sentence.replace(search_pattern, self.with_replacement)

        return sentence

    def remove_junkwords_from(self, sentence):
        for stopword in self._search_patterns:
            sentence = self._remove(sentence, stopword)

        return sentence


class TextParser(object):

    def __init__(self, corpus_path="data/Wikipedia-20190410064747.xml"):
        pass

    def read_raw_corpus(self):
        pass

    def save_proccessed_corpus(self):
        pass


wiki_corpus = "/home/alex/Dokumente/Wikipedia-20190410012444_np.xml"
class_labels = "/media/alex/Projects/Pilze_Fastai/DeVise/data/translate/6k_arten.txt"
page_splitter = "</page>"
tbox_marker = "{{Taxobox"
tbox_end_marker = "}}\n"
line_spliter = "\n"
sci_name_mrkr = "<title>"
sci_name_end = "</title>\n"
comm_name = "<redirect title=\""
comm_name_end = "\" />\n"
junk_chars = ["[", "]", "{", "}", "=", "'", "(", ")",
              "|", "&lt;", "&gt;", "*", "&amp;",
              "±", "&amp;nbsp;", "nbsp;", "&lt;ref",
              "/ref", "ref name\"", "name=", "!--",
              "--", "``", "–", "-", "/", "\\", "button",
              "|_taxon_name_______=", "</title>", "____",
              "~~~~", "_taxon_name_______="]
# nums = "1 2 3 4 5 6 7 8 9 0".split(" ")
spez_words = "wikipedia dewiki https wikipedia wiki wikipedia hauptseite mediawiki first letter medium spezial diskussion benutzer portal ____ wikitext text wiki weiterleitung".split(
    " ")
punctuation = [".", ",", ":", "%", "&", "!", "?", ";"]
junk_chars = junk_chars + spez_words  # + nums
sentence_corpus = []

sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle')
fungi_classes = set()

with open(class_labels, 'r') as cls_reader:
    fungi_lbls = cls_reader.readlines()

for lbl in fungi_lbls:
    genus = lbl.split('-')[0]
    fungi_classes.add(genus.lower())


def replace_umlauts(text):
    """
    Replaces german umlauts and sharp s in given text.

    :param text: text as str
    :return: manipulated text as str
    """
    res = text
    res = res.replace('ä', 'ae')
    res = res.replace('ö', 'oe')
    res = res.replace('ü', 'ue')
    res = res.replace('Ä', 'Ae')
    res = res.replace('Ö', 'Oe')
    res = res.replace('Ü', 'Ue')
    res = res.replace('ß', 'ss')
    return res


def remove_junk_chars(page, junk_chars):
    for char in junk_chars:
        page = page.replace(char, ' ')
    return page


def remove_punctuation(sentence):
    for symb in punctuation:
        sentence = sentence.replace(symb, ' ')
    return sentence


def remove_stopwords(sentences, sci_name="", comm_name=""):
    for sentence in sentences:
        sentence = remove_punctuation(sentence)
        words = nltk.word_tokenize(sentence, 'german')
        words = [x for x in words if x not in stop_words and not x.isdigit() and 5 < len(x) < 27]
        if len(words) > 2:
            sentence_corpus.append(words)
        else:
            words.append(sci_name)
            words.append(comm_name)
            sentence_corpus.append(words)

    return True


def read_corpus(path_corpus):
    with open(path_corpus, 'r') as wiki_corp:
        content = wiki_corp.read()

    return content


def split_artikles(plain_corpus):
    articles = plain_corpus.split(page_splitter)
    return articles


def parse_articles(articles):
    for article in articles:
        page_txt = replace_umlauts(article)
        parse_one_article(page_txt)


def safe_fungi_names(page_content):
    for genus in fungi_classes:
        tok = "{} ".format(genus)
        with_this_token = "{}_".format(genus)
        page_content = page_content.replace(tok, with_this_token)
    return page_content


def parse_one_article(page_content):
    tbox_lines = []
    sci_name = ""
    ger_common_name = ""
    is_tbox_data = False
    article_lines = page_content.split(line_spliter)
    if tbox_marker in page_content:

        for line in article_lines:
            if is_tbox_data:
                if "Taxon_Name" in line:
                    ger_common_name = line.split(" = ")[-1].replace("-",
                                                                    "_").replace(" ",
                                                                                 "_").lower()
                elif "Taxon_WissName" in line:
                    sci_name = line.split(" = ")[-1].replace("-",
                                                             "_").replace(" ",
                                                                          "_").lower()
                tbox_lines.append(line)

            if tbox_marker in line:
                is_tbox_data = True
            elif tbox_end_marker in line:
                is_tbox_data = False
    else:

        for line in article_lines:
            if sci_name_mrkr in line:
                sci_name = line.replace(sci_name_mrkr,
                                        '').replace(sci_name_end,
                                                    '').replace(" ",
                                                                "_").lower()
            elif comm_name in line:
                ger_common_name = line.replace(comm_name,
                                               '').replace(comm_name_end,
                                                           '').replace(" ",
                                                                       "_").replace("-",
                                                                                    '_').lower()
    if ger_common_name == "":
        ger_common_name = sci_name

    markdwn_free = filter_wiki(page_content).lower()
    page_lower = markdwn_free.replace(sci_name.replace('_', ' '),
                                      sci_name.replace(' ', '_'))
    page_lower = page_lower.replace(ger_common_name.replace('_', ' '),
                                    ger_common_name.replace(' ', '_'))
    page_ = remove_junk_chars(page_lower, junk_chars).split("einzelnachweise")[0]
    p_content = safe_fungi_names(page_)
    sentences = sentence_detector.tokenize(p_content)
    ok_ = remove_stopwords(sentences, sci_name, ger_common_name)


def save_corpus_to_file(corpus):
    dictionary = corpora.Dictionary(corpus)
    dictionary.save('fungi.dict')
    dictionary.save_as_text("fungi_emmbed.txt")
    mm_corpus = corpora.
    print(dictionary)


def write_txt_file(corpus):
    with open('fungi.txt', 'a') as writer:
        for sent in corpus:
            line = ' '.join(sent) + '\n'
            writer.write(line)


def display_closestwords_tsnescatterplot(model, word):
    arr = np.empty((0, 300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.most_similar(word)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=42)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()


stop_words = [replace_umlauts(token) for token in stopwords.words('german')]
content = read_corpus(wiki_corpus)
articles = split_artikles(content)
parse_articles(articles)
# save_corpus_to_file(sentence_corpus)
# write_txt_file(sentence_corpus)
model = FastText(size=300, window=7, min_count=2)  # instantiate
model.build_vocab(sentences=sentence_corpus)
print('Training')
model.train(sentences=sentence_corpus, total_examples=model.corpus_count, epochs=100)  # train
# fname = get_tmpfile("fasttext.model")
# model.save("fasttext.model")
print('Plotting')
display_closestwords_tsnescatterplot(model, "Giftpilze")
