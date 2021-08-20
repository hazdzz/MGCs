import numpy as np
import pandas as pd

def preprocess_webkb_data(dataset_name, dataset_path, graph_path):
    content = pd.read_csv(dataset_path, sep='\t', header=None)
    n_vertex = content.shape[0]
    index = list(content.index)
    content_np = content.to_numpy()
    content_np_ = np.zeros(n_vertex, dtype=object)
    for i in range(n_vertex):
        content_np_[i] = content_np[i][0]
    id_ = content_np_.tolist()
    map_ = dict(zip(id_, index))

    features = content.iloc[:,1:-1]
    np.savetxt('features.csv', features, fmt='%s', delimiter=',')

    labels = pd.get_dummies(content[content.shape[1]-1])
    np.savetxt('labels.csv', labels, fmt='%s', delimiter=',')

    cites = pd.read_csv(graph_path, sep=' ', header=None)
    cites = cites.to_numpy()

    cites_row = np.zeros(cites.shape[0], dtype=object)
    cites_col = np.zeros(cites.shape[0], dtype=object)
    for i in range(cites.shape[0]):
        cites_row[i] = cites[i][0]
        cites_col[i] = cites[i][1]

    cites_row_ = []
    for i in range(cites.shape[0]):
        if cites_row[i] in map_.keys():
            cites_row_.append(map_[cites_row[i]])

    cites_col_ = []
    for i in range(cites.shape[0]):
        if cites_col[i] in map_.keys():
            cites_col_.append(map_[cites_col[i]])

    dir_adj = np.zeros((n_vertex, n_vertex), dtype=int)
    for i in range(cites.shape[0]):
        dir_adj[cites_row_[i]][cites_col_[i]] += 1
    
    np.savetxt('dir_adj.csv', dir_adj, fmt='%s', delimiter=',')