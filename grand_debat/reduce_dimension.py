import umap
from bokeh.plotting import figure, output_file, show, ColumnDataSource
import pandas as pd
from matplotlib import colors as mcolors


def compute_low_dim_embeddings(data, min_dist=0.01, n_neighbors=15, metric='cosine'):
    return umap.UMAP(min_dist=min_dist, n_neighbors=n_neighbors, metric=metric, random_state=123).fit_transform(data)


def get_low_dim_embeddings_df(words, data, min_dist=0.01, n_neighbors=15, metric='cosine'):
    umap_embeddings = compute_low_dim_embeddings(data, min_dist, n_neighbors, metric)
    embeddings_df = pd.DataFrame(umap_embeddings, columns=["x", "y"]).join(pd.Series(words, name="word"))
    embeddings_df['embeddings'] = data
    return embeddings_df.loc[:, ["word", "embeddings", "x", "y"]]


def plot_embeddings(embeddings_df, output_path, title, plot_top_n_words=None):
    if plot_top_n_words is not None:
        plot_df = embeddings_df.head(plot_top_n_words)
    else:
        plot_df = embeddings_df
    output_file(output_path)

    source = ColumnDataSource(data=dict(
        x=plot_df["x"].values.tolist(),
        y=plot_df["y"].values.tolist(),
        desc=plot_df["word"].values.tolist(),
    ))

    TOOLTIPS = [
        ("desc", "@desc"),
    ]

    p = figure(plot_width=500, plot_height=500, tooltips=TOOLTIPS,
               title=title)

    p.circle('x', 'y', size=3, source=source, color="navy", alpha=0.2)

    show(p)
    return


def plot_embeddings_with_clusters(embeddings_df, output_path, title, top_n=10, clusters=None):
    output_file(output_path)
    colors = list(mcolors._colors_full_map.values())
    if len(clusters) > 0:
        TOOLTIPS = [
            ("desc", "@desc"),
        ]
        p = figure(plot_width=500, plot_height=500, tooltips=TOOLTIPS,
                   title=title)

        sources = []
        for cluster in clusters:
            sources.append(ColumnDataSource(data=dict(
                x=embeddings_df[embeddings_df["cluster"] == cluster].head(top_n)["x"].values.tolist(),
                y=embeddings_df[embeddings_df["cluster"] == cluster].head(top_n)["y"].values.tolist(),
                desc=embeddings_df[embeddings_df["cluster"] == cluster].head(top_n)["word"].values.tolist(),
            )))

        for i, source in enumerate(sources):
            p.circle('x', 'y', size=3, source=source, color=colors[i], alpha=0.3)

        show(p)

    else:
        plot_embeddings(embeddings_df, output_path, title)
    return
