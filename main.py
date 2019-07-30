from grand_debat import collect_and_load
from grand_debat import process
from grand_debat import modeling
from grand_debat import reduce_dimension
from grand_debat import clustering
from grand_debat import pagerank
import json
from nltk.corpus import stopwords
import os
import re

# Load conf with URL of raw csv files and closed questions list per file
with open("conf.json", 'rb') as json_file:
    conf = json.load(json_file)

# Define path
DATA_FOLDER = "data"
MODEL_FOLDER = "model"
PLOT_FOLDER = "viz"

# Create path
for folder in [DATA_FOLDER, MODEL_FOLDER, PLOT_FOLDER]:
    if os.path.exists(os.getcwd() + "/" + folder) == False:
        os.makedirs(os.getcwd() + "/" + folder + "/")

#Parameters
# IDCOLS are columns that are not contributions from citizens
IDCOLS = ['authorId', 'authorType', 'authorZipCode', 'createdAt', 'id', 'publishedAt', 'reference', 'title', 'trashed',
          'trashedStatus', 'updatedAt']

#Liste of stop words
STOPWORDS = set(
    stopwords.words('french') +
    ["none", "les", "etc", "leurs", "ils", "car", "cela",
     "cet", "cette", "ou", "or", "neant", "nsp"]
)

# Do you wanna plot low dim embeddings ?
PLOTS = False

# Max number of words used
TOP_N_WORDS = 8000

# SEE THEMES LINKED TO KEYWORDS
KEYWORDS = ["ecologie", "impot", "sante", "politique", "president"]


def main(keywords):
    # Collect data - Download all raw csv files in data folder
    collect_and_load.collect_all_files(conf, folder=DATA_FOLDER)

    # Process data - Clean text data & build contributions.txt file
    if os.path.exists(os.getcwd() + "/" + DATA_FOLDER + "/" + "contributions.txt") == False:
        # Load all files - drop closed questions
        df = collect_and_load.load_csv_files(conf, folder=DATA_FOLDER)

        # Concatenate and process questions columns
        questions_cols = [col for col in df.columns if col not in IDCOLS]
        df = df.loc[:, questions_cols].fillna("")


        raw_contributions = [df[df[q_col].map(len) > 0].loc[:, q_col].values.tolist() for q_col in questions_cols]

        # Flatten contribs & remove duplicates
        raw_contributions = list(set(
            [item for sublist in raw_contributions for item in
             sublist]))

        raw_contributions = [re.sub("[#\"$’%&\'()'*+,\-/:;<-=>@[\\]^_`{|}~]", ' ', re.sub("€", " euros", contrib)) for contrib
                             in raw_contributions]

        raw_contributions = [contrib for contrib in raw_contributions if len(contrib.split(" "))>0]
        # Clean contributions & save results in contributions.txt
        process.clean_contributions(raw_contributions,
                                    corpus_path=os.getcwd() + "/" + DATA_FOLDER + "/contributions.txt",
                                    min_word_len=1,
                                    stop_words=STOPWORDS)

    else:
        print("contributions.txt already exists.")

    # COMPUTE EMBEDDINGS & REDUCE DIM TO PLOT
    # Train fastText model
    if os.path.exists(os.getcwd() + "/" + MODEL_FOLDER + "/model_gd") == False:
        model = modeling.train_model(data_path=os.getcwd() + "/" + DATA_FOLDER + "/contributions.txt",
                                     size_embeddings=64, epochs=64)
        model.save(os.getcwd() + "/" + MODEL_FOLDER + "/model_gd")
    else:
        print("model_gd already exists.")

    # Load model
    model = modeling.load_model(os.getcwd() + "/" + MODEL_FOLDER + "/model_gd")

    # Get words embeddings
    words, embeddings = modeling.get_top_words_embeddings(model, top_n_words=TOP_N_WORDS)

    # Reduce Dimension of embeddings using UMAP - return a pandas df
    low_dim_embeddings_df = reduce_dimension.get_low_dim_embeddings_df(words, embeddings)

    # Plot embeddings
    if PLOTS:
        reduce_dimension.plot_embeddings(low_dim_embeddings_df,
                                         output_path=os.getcwd() + "/" + PLOT_FOLDER + "/word_embeddings_top_" + str(
                                             TOP_N_WORDS) + ".html",
                                         title="Embeddings des " + str(TOP_N_WORDS) + " mots les plus fréquents",
                                         plot_top_n_words=TOP_N_WORDS)

    # CLUSTERING : FIND THEMES BY CLUSTERING EMBEDDINGS
    # UNCOMMENT TO USE Kmeans
    # clustered_low_dim_embeddings_df = clustering.compute_and_get_kmeans_clusters(low_dim_embeddings_df, number_of_clusters=100)

    # Hierarchical clustering
    # To get a viz of the dendrogram add an output_path parameter such as
    # output_path=os.getcwd() + "/" + PLOT_FOLDER + "/hierarchical_clustering.pdf"
    clustered_low_dim_embeddings_df = clustering.compute_and_get_hierarchical_clusters(low_dim_embeddings_df,
                                                                                       number_of_clusters=75)

    clusters_for_keywords = clustered_low_dim_embeddings_df[
        clustered_low_dim_embeddings_df["word"].isin(keywords)].groupby("cluster")["word"].agg(list).to_dict()

    if PLOTS:
        reduce_dimension.plot_embeddings_with_clusters(clustered_low_dim_embeddings_df,
                                                       output_path=os.getcwd() + "/" + PLOT_FOLDER + "/test.html",
                                                       title="Zoom sur 2 clusters",
                                                       clusters=clusters_for_keywords.keys())

    # add page rank score
    clustered_low_dim_embeddings_df = pagerank.get_page_rank_df(clustered_low_dim_embeddings_df)


    for cluster in clusters_for_keywords.keys():
        print("*" * 50)
        print("Cluster keywords :", clusters_for_keywords[cluster])
        print(clustered_low_dim_embeddings_df[clustered_low_dim_embeddings_df["cluster"] == cluster].sort_values(
            "pagerank_score", ascending=False).head(10).loc[:, "word"].values.tolist())


if __name__ == '__main__':
    main(KEYWORDS)
