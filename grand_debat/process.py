import unicodedata
import numpy as np
import pandas as pd
import re
import swifter
import nltk
import string
import spacy
from spacy_lefff import LefffLemmatizer, POSTagger

nlp = spacy.load('fr_core_news_sm', disable=['tagger', 'ner'])
punctuations = string.punctuation

pos = POSTagger()
french_lemmatizer = LefffLemmatizer(after_melt=True)  #after_melt=True
nlp.add_pipe(pos, name='pos', after='parser')
nlp.add_pipe(french_lemmatizer, name='lefff', after='pos')  # , after='pos'

def remove_punctuation(doc, punctuations = punctuations):
    out_doc = doc.text
    for punc in punctuations:
        out_doc = out_doc.replace(punc, " ")
    return spacy.make_doc(out_doc)

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


def clean_contributions(raw_contributions, corpus_path, min_word_len=2, stop_words=None):
    with open(corpus_path, "w") as f:
        for doc in nlp.pipe(raw_contributions, batch_size=100000, n_threads=8):
            for sent in doc.sents:
                out_sent = []
                for tok in sent:
                    if not tok.is_punct:
                        if tok._.lefff_lemma is None:
                            out_sent.append(remove_accents(tok.lemma_.lower().strip()))
                        else:
                            out_sent.append(remove_accents(tok._.lefff_lemma.lower().strip()))

                out_sent = [tok for tok in out_sent if tok not in stop_words and len(tok) > min_word_len]
                if len(out_sent) > 1:
                    # Write only if len(sentence) > 1 word
                    f.write(" ".join(out_sent).strip() + "\n")


def lemmatize(sentences):
    res = []
    for doc in nlp.pipe(sentences, batch_size=1000):
        curRes = []
        try:
            for d in doc:
                if d._.lefff_lemma is None:
                    curRes.append(d.lemma_)
                else:
                    curRes.append(d._.lefff_lemma)
        except:
            curRes = doc
        res.append(u" ".join(curRes))
    return res


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


def remove_punctuation(sentence):
    pattern = re.compile('\W+')
    return re.sub(pattern, ' ', sentence).strip()


def process_text(sentence, stop_words, min_word_len=3):
    if len(sentence) >= min_word_len:
        res = (remove_accents(remove_punctuation(sentence)), stop_words,
               min_word_len)
        if len(res) > 0:
            return (res + " | ").strip()
        else:
            return ""
    else:
        return ""


def process_text_cols(row, stop_words, min_word_len=3):
    return u"".join([process_text(rowc, stop_words, min_word_len) for rowc in row if rowc is not None])


def build_text_col_parallel(df, stop_words, min_word_len=3, output_colname="TextFeatures"):
    df.loc[:, output_colname] = df.swifter.progress_bar(True).apply(
        lambda x: process_text_cols(x, stop_words, min_word_len), axis=1)
    df.loc[:, output_colname] = df.loc[:, output_colname].map(
        lambda x: np.array([sentence.strip() for sentence in x.split("|") if len(sentence) > 0]))
    return df


def save_sentence_corpus(df, corpus_path):
    print("Saving contributions.txt")
    with open(corpus_path, "w") as f:
        # Flatten sentence list
        for contrib in df[df["TextFeatures"].map(len) > 0].loc[:, "TextFeatures"].values:
            for sentence in contrib:
                f.write(sentence + "\n")
    return
