from abc import abstractmethod, ABCMeta

import nltk.data
from gensim.corpora.wikicorpus import filter_wiki
from gensim import corpora
from nltk.corpus import stopwords
import numpy as np

sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle')

default_umlaut_search = "ä ö ü Ä Ö Ü ß".split(' ')
default_umlaut_replacement = "ae oe ue Ae Oe Ue ss".split(' ')
default_corpora_path = "data/Wikipedia-20190410064747.xml"
default_corpora_type = "WikiFungi"

DEFAULT = " "

wiki_corpus = "data/Wikipedia-20190410064747.xml"
proccessed_corpora_path = "proccessed_data/{}".format(default_corpora_type + '{}')
class_labels = "data/articles_6k_classes"

page_splitter = "</page>"
tbox_marker = "{{Taxobox"
tbox_end = "}}\n"
line_spliter = "\n"
sci_name_mrkr = "<title>"
sci_name_end = "</title>\n"
comm_name = "<redirect title=\""
comm_name_end = "\" />\n"

punctuation = [".", ",", ":", "%", "&", "!", "?", ";"]
junk_txts = ["[", "]", "{", "}", "=", "'", "(", ")",
             "|", "&lt;", "&gt;", "*", "&amp;",
             "±", "&amp;nbsp;", "nbsp;", "&lt;ref",
             "/ref", "ref name\"", "name=", "!--",
             "--", "``", "–", "-", "/", "\\", "button",
             "|_taxon_name_______=", "</title>", "__",
             "~~~~", "_taxon_name_______=", "</title> <redirect title=\"", "___"]
nums = list(range(0 , 2100)) #"1 2 3 4 5 6 7 8 9 0".split(" ")
spez_words = ["wikipedia", "dewiki", "https", "wikipedia", "wiki", "wikipedia",
              "hauptseite", "mediawiki", "first", "letter", "medium", "spezial",
              "diskussion", "benutzer", "portal", "____", "wikitext", "text", "wiki", "weiterleitung"]

junk_txts = junk_txts + spez_words  # + nums

sentence_corpus = []
fungi_classes = set()

with open(class_labels, 'r') as cls_reader:
    fungi_lbls = cls_reader.readlines()

for lbl in fungi_lbls:
    genus = lbl.split('-')[0]
    fungi_classes.add(genus.lower())


class Cleaner(metaclass=ABCMeta):

    def __init__(self, search_patterns, replacement=' '):
        self._search_patterns = search_patterns
        if replacement is not DEFAULT:
            self.with_replacement = replacement
        else:
            self.with_replacement = ' '

    @abstractmethod
    def _remove(self, sentence, search_pattern, **kwargs):
        """
        Deletes search_pattern inside sentence
        :param sentence:
        :param search_pattern: pattern which to delete or replace
        :param kwargs: could be replacements
        """
        pass


class Punctuation(Cleaner):

    def __init__(self, target_punctuation, replacement):
        super().__init__(search_patterns=target_punctuation, replacement=replacement)

    def _remove(self, sentence, search_pattern, **kwargs):
        sentence = sentence.replace(search_pattern, self.with_replacement)

        return sentence

    def remove_punctuation_from(self, sentence):
        for punct_ in self._search_patterns:
            sentence = self._remove(sentence=sentence, search_pattern=punct_)

        return sentence


class Umlaut(Cleaner):

    def __init__(self, search_patterns, replacement):
        if search_patterns is None:
            super().__init__(default_umlaut_search, default_umlaut_replacement)
        else:
            super().__init__(search_patterns, replacement)

    def _remove(self, page_text, search_pattern, **kwargs):
        """
        Replaces german umlauts and sharp s in given text.

        :param text: text as str
        :return: manipulated text as str
        """
        res = page_text
        res = res.replace('ä', 'ae')
        res = res.replace('ö', 'oe')
        res = res.replace('ü', 'ue')
        res = res.replace('Ä', 'Ae')
        res = res.replace('Ö', 'Oe')
        res = res.replace('Ü', 'Ue')
        res = res.replace('ß', 'ss')
        return res

    def remove_umlauts_from(self, page):
        #for umlaut, replacement in zip(self._search_patterns, self.with_replacement):
        page = self._remove(page, search_pattern="ä", kwargs="ae")

        return page


class Stopwords(Cleaner):
    def __init__(self, search_patterns, replacement=' '):
        super().__init__(search_patterns, replacement)

    def _remove(self, sentence, search_pattern, **kwargs):
        sentence = sentence.replace(str(search_pattern), self.with_replacement)

        return sentence

    def remove_stopwords_from(self, sentence):
        for stopword in self._search_patterns:

            sentence = self._remove(sentence, stopword)

        return sentence


class JunkText(Cleaner):
    def __init__(self, search_patterns, replacement=' '):
        super().__init__(search_patterns, replacement)

    def _remove(self, sentence, search_pattern, **kwargs):
        sentence = sentence.replace(search_pattern, self.with_replacement)

        return sentence

    def remove_junkwords_from(self, sentence):
        for junk_txt in self._search_patterns:
            sentence = self._remove(sentence, junk_txt)

        return sentence


class Parser(metaclass=ABCMeta):

    def __init__(self, corpora_path, name):
        self._src_path = corpora_path
        self._parser_type = name
        self._fname = name + '{}'

    @abstractmethod
    def _read_raw_corpora(self):
        pass

    @abstractmethod
    def _save_dictionary(self, corpora, fname, as_txt=False):
        pass

    @abstractmethod
    def _save_corpora(self, corpora, fname):
        pass


class Text(Parser):

    def __init__(self, corpora_path="data/Wikipedia-20190410064747.xml", name="WikiFungi"):
        super().__init__(corpora_path, name)
        self._content = self._read_raw_corpora()

    def _read_raw_corpora(self):
        """
        Read raw Wiki_xxxx.xml file as one big string
        :return corp_content: .xml file content
        """
        try:
            with open(self._src_path, 'r') as corpora:
                corp_content = corpora.read()

            return corp_content
        except FileNotFoundError:
            print("Corpus file NOT found !")
            raise FileNotFoundError

    def _save_dictionary(self, corpora_, fname, as_txt=False):
        """
        Save corpora dictionary
        :param corpora_:
        :param fname: "filename{}" pattern
        :param as_txt: should save as .txt file
        :return:
        """
        dictionary = corpora.Dictionary(corpora_)
        try:
            if as_txt:
                #f_name = fname.format('.txt')
                dictionary.save_as_text(fname)
            else:
                #f_name = fname.format('.dict')
                dictionary.save(fname)
            return True
        except Exception:
            print("Dictionary NOT saved !")
            raise Exception

    def _save_corpora(self, corpora_, fname):
        """
        Save wiki corpora as .txt file
        :param corpora_:
        :param fname:
        :return:
        """
        try:
            with open(fname, 'a') as writer:
                for sent in corpora_:
                    line = ' '.join(sent) + '\n'
                    writer.write(line)
            return True
        except Exception:
            print("Corpora not saved")
            raise Exception


class Wiki(Text):

    def __init__(self):
        super().__init__(default_corpora_path, default_corpora_type)
        self._punctuation_cleaner = Punctuation(punctuation, replacement=' ')
        self._umlaut_cleaner = Umlaut(default_umlaut_search, default_umlaut_replacement)
        uml_cleaned = [self._umlaut_cleaner.remove_umlauts_from(token) for token in stopwords.words('german')]
        self._stop_words = uml_cleaned + nums
        self._stopword_cleaner = Stopwords(self._stop_words, " ")
        self._junk_txt_cleaner = JunkText(junk_txts, " ")
        self._proccessed_corpus_sentences = []

    def save(self, corpora_, as_text=False):
        """
        depricated maybe
        :param corpora_:
        :param as_text:
        """
        dict_fname = self._fname.format('_dictionary.txt')
        corpora_fname = self._fname.format('_corpora.txt')
        if as_text:
            self._save_dictionary(corpora_, dict_fname, as_txt=True)
            self._save_corpora(corpora_, fname=corpora_fname)
        else:
            self._save_dictionary(corpora_, self._fname.format('_dictionary.dict'))

        return dict_fname, corpora_fname

    def split_articles(self, plain_corpora):
        """
        Split whole .txt file into wiki articles
        :param plain_corpora: plain corpora .txt file content as str
        :return articles: list of wiki articles
        """
        articles_ = plain_corpora.split(page_splitter)
        return articles_

    def proccess_articles(self, articles_):
        """
        Cleans all given articles
        :param articles_: list of wiki articles
        """
        for article in articles_:
            page_txt = self._umlaut_cleaner.remove_umlauts_from(article)
            self.proccess_one_article(page_txt)

    def proccess_one_article(self, page_content):
        """
        Cleans one wiki article
        :param page_content: wiki article content
        """
        tbox_lines = []
        sci_name = ""
        ger_common_name = ""
        is_tbox_data = False
        article_lines = page_content.split(line_spliter)
        if tbox_marker in page_content:
            # Taxobox beginning
            for line in article_lines:
                if is_tbox_data:
                    # collect scientific and common fungi names
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
                elif tbox_end in line:
                    is_tbox_data = False
        else:
            # else just collect scientific and common name
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

        # markdwn_free = filter_wiki(page_content).lower()
        lower_case_content = page_content.lower()
        page_lower = lower_case_content.replace(sci_name.replace('_', ' '),
                                          sci_name.replace(' ', '_'))
        page_lower = page_lower.replace(ger_common_name.replace('_', ' '),
                                        ger_common_name.replace(' ', '_'))
        page_content = self._junk_txt_cleaner.remove_junkwords_from(page_lower).split("einzelnachweise")[0]
        page_content = self.combine_fungiclasses_consisting_of_several_words(page_content)
        sentences = sentence_detector.tokenize(page_content)

        for sentence in sentences:
            sentence_ = self._punctuation_cleaner.remove_punctuation_from(sentence)
            sentence_ = self._stopword_cleaner.remove_stopwords_from(sentence_)
            words = nltk.word_tokenize(sentence_, 'german')
            words = [x.lower() for x in words if not x.isdigit() and 5 < len(x) < 27]
            if len(words) > 2:
                self._proccessed_corpus_sentences.append(words)
            else:
                words.append(sci_name)
                words.append(comm_name)
                self._proccessed_corpus_sentences.append(words)

        return True

    def combine_fungiclasses_consisting_of_several_words(self, page_content):
        """
        Combine Fungi Class names consisting of several words  with an underscore '_'
        :rtype: str
        :param page_content:
        :return page_content:
        """
        for genus in fungi_classes:
            tok = "{} ".format(genus)
            with_new_token = "{}_".format(genus)
            page_content = page_content.replace(tok, with_new_token)
        return page_content

    def clean_corpora(self, should_save=False):
        d_fname, corpora_fname = ["", ""]
        #mdown_free_content = filter_wiki(self._content)
        articles_ = self.split_articles(self._content)
        for article in articles_:
            self.proccess_one_article(article)

        if should_save:
            d_fname, corpora_fname = self.save(self._proccessed_corpus_sentences, as_text=True)

        return self._proccessed_corpus_sentences, d_fname, corpora_fname

# def replace_umlauts(text):
#     """
#     Replaces german umlauts and sharp s in given text.
#
#     :param text: text as str
#     :return: manipulated text as str
#     """
#     res = text
#     res = res.replace('ä', 'ae')
#     res = res.replace('ö', 'oe')
#     res = res.replace('ü', 'ue')
#     res = res.replace('Ä', 'Ae')
#     res = res.replace('Ö', 'Oe')
#     res = res.replace('Ü', 'Ue')
#     res = res.replace('ß', 'ss')
#     return res
#
#
# def remove_junk_chars(page, junk_chars):
#     for char in junk_chars:
#         page = page.replace(char, ' ')
#     return page
#
#
# def remove_punctuation(sentence):
#     for symb in punctuation:
#         sentence = sentence.replace(symb, ' ')
#     return sentence
#
#
# def remove_stopwords(sentences, sci_name="", comm_name=""):
#     for sentence in sentences:
#         sentence = remove_punctuation(sentence)
#         words = nltk.word_tokenize(sentence, 'german')
#         words = [x for x in words if x not in stop_words and not x.isdigit() and 5 < len(x) < 27]
#         if len(words) > 2:
#             sentence_corpus.append(words)
#         else:
#             words.append(sci_name)
#             words.append(comm_name)
#             sentence_corpus.append(words)
#
#     return True
#
#
# def read_corpus(path_corpus):
#     with open(path_corpus, 'r') as wiki_corp:
#         content = wiki_corp.read()
#
#     return content
#
#
# def split_artikles(plain_corpus):
#     articles = plain_corpus.split(page_splitter)
#     return articles
#
#
# def parse_articles(articles):
#     for article in articles:
#         page_txt = replace_umlauts(article)
#         parse_one_article(page_txt)
#
#
# def safe_fungi_names(page_content):
#     for genus in fungi_classes:
#         tok = "{} ".format(genus)
#         with_this_token = "{}_".format(genus)
#         page_content = page_content.replace(tok, with_this_token)
#     return page_content
#
#
# def parse_one_article(page_content):
#     tbox_lines = []
#     sci_name = ""
#     ger_common_name = ""
#     is_tbox_data = False
#     article_lines = page_content.split(line_spliter)
#     if tbox_marker in page_content:
#
#         for line in article_lines:
#             if is_tbox_data:
#                 if "Taxon_Name" in line:
#                     ger_common_name = line.split(" = ")[-1].replace("-",
#                                                                     "_").replace(" ",
#                                                                                  "_").lower()
#                 elif "Taxon_WissName" in line:
#                     sci_name = line.split(" = ")[-1].replace("-",
#                                                              "_").replace(" ",
#                                                                           "_").lower()
#                 tbox_lines.append(line)
#
#             if tbox_marker in line:
#                 is_tbox_data = True
#             elif tbox_end in line:
#                 is_tbox_data = False
#     else:
#
#         for line in article_lines:
#             if sci_name_mrkr in line:
#                 sci_name = line.replace(sci_name_mrkr,
#                                         '').replace(sci_name_end,
#                                                     '').replace(" ",
#                                                                 "_").lower()
#             elif comm_name in line:
#                 ger_common_name = line.replace(comm_name,
#                                                '').replace(comm_name_end,
#                                                            '').replace(" ",
#                                                                        "_").replace("-",
#                                                                                     '_').lower()
#     if ger_common_name == "":
#         ger_common_name = sci_name
#
#     markdwn_free = filter_wiki(page_content).lower()
#     page_lower = markdwn_free.replace(sci_name.replace('_', ' '),
#                                       sci_name.replace(' ', '_'))
#     page_lower = page_lower.replace(ger_common_name.replace('_', ' '),
#                                     ger_common_name.replace(' ', '_'))
#     page_ = remove_junk_chars(page_lower, junk_txts).split("einzelnachweise")[0]
#     p_content = safe_fungi_names(page_)
#     sentences = sentence_detector.tokenize(p_content)
#     ok_ = remove_stopwords(sentences, sci_name, ger_common_name)
#
#
# def save_corpus_to_file(corpus):
#     dictionary = corpora.Dictionary(corpus)
#     dictionary.save('fungi.dict')
#     dictionary.save_as_text("fungi_emmbed.txt")
#     # mm_corpus = corpora.
#     print(dictionary)
#
#
# def write_txt_file(corpus):
#     with open('fungi.txt', 'a') as writer:
#         for sent in corpus:
#             line = ' '.join(sent) + '\n'
#             writer.write(line)
#
#
# def display_closestwords_tsnescatterplot(model, word):
#     arr = np.empty((0, 300), dtype='f')
#     word_labels = [word]
#
#     # get close words
#     close_words = model.most_similar(word)
#
#     # add the vector for each of the closest words to the array
#     arr = np.append(arr, np.array([model[word]]), axis=0)
#     for wrd_score in close_words:
#         wrd_vector = model[wrd_score[0]]
#         word_labels.append(wrd_score[0])
#         arr = np.append(arr, np.array([wrd_vector]), axis=0)
#
#     # find tsne coords for 2 dimensions
#     tsne = TSNE(n_components=2, random_state=42)
#     np.set_printoptions(suppress=True)
#     Y = tsne.fit_transform(arr)
#
#     x_coords = Y[:, 0]
#     y_coords = Y[:, 1]
#     # display scatter plot
#     plt.scatter(x_coords, y_coords)
#
#     for label, x, y in zip(word_labels, x_coords, y_coords):
#         plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
#     plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
#     plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
#     plt.show()
#
#
# stop_words = [replace_umlauts(token) for token in stopwords.words('german')]
# content = read_corpus(wiki_corpus)
# articles = split_artikles(content)
# parse_articles(articles)
# # save_corpus_to_file(sentence_corpus)
# # write_txt_file(sentence_corpus)
# model = FastText(size=300, window=7, min_count=2)  # instantiate
# model.build_vocab(sentences=sentence_corpus)
# print('Training')
# model.train(sentences=sentence_corpus, total_examples=model.corpus_count, epochs=100)  # train
# # fname = get_tmpfile("fasttext.model")
# # model.save("fasttext.model")
# print('Plotting')
# display_closestwords_tsnescatterplot(model, "Giftpilze")
