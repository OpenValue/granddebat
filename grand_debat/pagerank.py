import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compute_page_rank(embedding_df):
    scores_dict = {}
    for cluster in embedding_df.loc[:, "cluster"].unique():
        cM = cosine_similarity(
            np.stack(embedding_df[embedding_df["cluster"] == cluster].loc[:, "embeddings"].values))
        cos_sim_mat = cM + 1

        nx_graph = nx.from_numpy_array(cos_sim_mat)
        scores_dict[cluster] = dict(zip(list(embedding_df[embedding_df["cluster"] == cluster].loc[:, "word"].values), \
                                        list(nx.pagerank_numpy(nx_graph).values())))

    return scores_dict

def get_page_rank_df(embedding_df):
    scores_dict = compute_page_rank(embedding_df)
    embedding_df.loc[:, "pagerank_score"] = embedding_df.apply(
        lambda x: scores_dict[x["cluster"]][x["word"]], axis=1)

    return embedding_df
