import unicodedata
import string
import spacy
from spacy_lefff import LefffLemmatizer, POSTagger

nlp = spacy.load('fr_core_news_sm', disable=['tagger', 'ner'])
punctuations = string.punctuation

pos = POSTagger()
french_lemmatizer = LefffLemmatizer(after_melt=True)
nlp.add_pipe(pos, name='pos', after='parser')
nlp.add_pipe(french_lemmatizer, name='lefff', after='pos')

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
