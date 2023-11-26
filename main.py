import warnings
import gensim
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from collections import defaultdict
import torch
from tkinter import Tk, Canvas, Button
import networkx as nx

# Suppress UserWarnings from gensim
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

def load_word_vectors(model_path, vocab):
    # Load pre-trained word vectors using gensim
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

    wordvec = []
    vocab_new = []

    # Extract word vectors for the given vocabulary
    for x in vocab:
        try:
            wordvec.append(model.get_vector(x))
            vocab_new.append(x)
        except KeyError:
            print(f"Ignoring {x}")

    return wordvec, vocab_new

def cluster_and_plot(wordvec, vocab_new, n_clusters=4):
    # Perform clustering using MiniBatchKMeans
    kmeans_model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=8, verbose=1, max_iter=30)
    kmeans_model.fit(wordvec)

    cluster_labels = kmeans_model.labels_
    cluster_to_words = defaultdict(list)
    
    # Group words by cluster
    for cluster_id, word in zip(cluster_labels, vocab_new):
        cluster_to_words[cluster_id].append(word)

    for words in cluster_to_words.values():
        print(words)

    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    wordvec_r = pca.fit_transform(wordvec)

    # Plot the results
    fig, ax = plt.subplots()

    for i, label in enumerate(kmeans_model.labels_):
        color = 'red' if label == 0 else 'blue' if label == 1 else 'green' if label == 2 else 'orange'
        ax.scatter(wordvec_r[i, 0], wordvec_r[i, 1], c=color)
        ax.annotate(vocab_new[i], xy=(wordvec_r[i, 0], wordvec_r[i, 1]), size=8)

    plt.show()

def create_graph():
    # Create a simple undirected graph using NetworkX
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    
    # Draw the graph with labels
    nx.draw(G, with_labels=True)
    plt.show()

def main():
    # Example usage
    model_path = 'path_to_your_binary_model_file.bin'
    vocabulary = ["斉藤", "リンゴ", "シマウマ", "東京", "ライオン", "名古屋", "ミカン", "ウシ", "メロン", "田中", "横浜", "鈴木"]

    word_vectors, vocabulary_new = load_word_vectors(model_path, vocabulary)
    cluster_and_plot(word_vectors, vocabulary_new, n_clusters=4)

    # Create a simple Tkinter window with a button to visualize a graph
    root = Tk()
    root.title("NetworkX Graph Visualization")

    button = Button(root, text="Visualize Graph", command=create_graph)
    button.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
