## Create env

    conda env create -f environment.yml  
    
## Activate env

    conda activate granddebat

## Download Spacy fr model

    python -m spacy download fr_core_news_sm


## Edit lefff.py

lefff.py file is located here: 
~/anaconda/envs/granddebat/lib/python3.6/site-packages/spacy_lefff/lefff.py

Replace __call\__ method by :

    def __call__(self, doc):      
        for token in doc:
            if token._.melt_tagger is not None:
                t = token._.melt_tagger.lower() if self.after_melt else token.pos_
            else:
                t = token.pos_
            lemma = self.lemmatize(token.text, t)
            token._.lefff_lemma = lemma
    return doc

## If needed, resolve bug 
if locale pb, then edit .bash_profile and add :

    export LC_ALL=en_US.UTF-8
    export LANG=en_US.UTF-8

then run 
    
    source ~/.bash_profile`

relaunch : 

    python -m spacy download fr_core_news_sm