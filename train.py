import matplotlib.pyplot as plt
# uncomment if gensim is installed
# !pip install gensim
# Need the interactive Tools for Matplotlib
import numpy as np
from gensim.models import Word2Vec
from gensim.scripts import word2vec2tensor
from sklearn.manifold import TSNE

from wiki_parser import *


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


def main():
    save_name = "word2vec.model"
    wiki_parser = Wiki()
    sentence_corpus_, d_fname, corpora_fname = wiki_parser.clean_corpora(should_save=True)
    model = Word2Vec(sentence_corpus_, size=150, window=5, min_count=5)
    # model = FastText(size=300, window=4, min_count=4)  # instantiate
    #model.build_vocab(sentences=sentence_corpus_)
    print('Training')
    model.train(sentences=sentence_corpus_, total_examples=model.corpus_count, epochs=50,
                total_words=model.corpus_total_words)  # train
    # fname = get_tmpfile("fasttext.model")
    model.wv.save_word2vec_format(save_name, binary=True)
    print('Plotting')
    # display_closestwords_tsnescatterplot(model, "amanita_muscaria")
    print(save_name)

    word2vec2tensor.word2vec2tensor(save_name, "fungi_w2v.tsv")


main()
