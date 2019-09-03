# GRAND DEBAT

Here is the source code linked to the [medium article] (https://medium.com/@contact_8561/une-analyse-des-donn%C3%A9es-du-grand-d%C3%A9bat-national-761f8b6e9e19)


# Instructions 
    git clone https://github.com/OpenValue/granddebat.git
## Create conda env
    cd granddebat
    conda env create -f environment.yml  
## Activate conda env
    conda activate granddebat
## Download Spacy fr model
    python -m spacy download fr_core_news_sm   
### (Optional) If needed, resolve bug 
if locale pb, then edit .bash_profile and add :

    export LC_ALL=en_US.UTF-8
    export LANG=en_US.UTF-8
then run 

    source ~/.bash_profile
launch : 

    python -m spacy download fr_core_news_sm

## RUN
If needed, change parameters of application : folder paths (for data, model and plots), stopwords list etc... (see CAPITALIZED variables in main.py).
Then run : 

    python main.py
    
