from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath


def train_model(data_path, size_embeddings, epochs=64):
    corpus_file = datapath(data_path)
    model_gensim = FT_gensim(size=size_embeddings, workers=4)
    # build the vocabulary
    model_gensim.build_vocab(corpus_file=corpus_file)
    # train the model
    model_gensim.train(
        corpus_file=corpus_file, epochs=epochs,
        total_examples=model_gensim.corpus_count, total_words=model_gensim.corpus_total_words
    )
    return model_gensim


def load_model(model_path):
    model = FT_gensim.load(model_path, mmap='r')
    return model


def get_top_words_embeddings(model, top_n_words):
    if top_n_words > len(model.wv.index2entity):
        words = model.wv.index2entity
    elif top_n_words <= 0:
        words = model.wv.index2entity
    else:
        words = model.wv.index2entity[:top_n_words]

    embeddings = [model.wv[word] for word in words]
    return words, embeddings


def get_words_embeddings(model):
    words = model.wv.index2entity
    embeddings = [model.wv[word] for word in words]
    return words, embeddings
