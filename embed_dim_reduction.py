'''
Perform dimensionality reduction on the word vectors
'''

import sys
import types

import torch.nn as nn
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
import ujson as json #pylint:disable=import-error

import util

def pca_dim_reduce():
    '''Reduce the word vector dimensions by half using PCA'''
    print('Loading word vectors...')
    w_embed = util.torch_from_json('./data/word_emb.json')
    new_embed_len = w_embed.shape[1] // 2
    print(f'Performing PCA to reduce dimensions from {w_embed.shape[1]} to {new_embed_len}...')
    pca = PCA(n_components=new_embed_len)
    res = pca.fit_transform(w_embed)
    output_f = f'./data/word_emb_pca_reduce_{new_embed_len}.json'
    print(f'Dumping json to {output_f}...')
    with open(output_f, 'w') as json_out:
        json.dump(res.tolist(), json_out)
    print('Success!')

def isomap_dim_reduce():
    '''
    Default args for ISOMAP:
    n_neighbors=5, n_components=2, eigen_solver='auto', tol=0, max_iter=None, path_method='auto',
    neighbors_algorithm='auto', n_jobs=None, metric='minkowski', p=2, metric_params=None

    https://towardsdatascience.com/dimensionality-reduction-toolbox-in-python-9a18995927cd
    '''
    print('Loading word vectors...')
    w_embed = util.torch_from_json('./data/word_emb.json')
    new_embed_len = w_embed.shape[1] // 2
    print(f'Performing ISOMAP to reduce dimensions from {w_embed.shape[1]} to {new_embed_len}...')
    isomap = Isomap(n_components=new_embed_len, n_jobs=4, n_neighbors=3)
    res = isomap.fit_transform(w_embed)
    output_f = f'./data/word_emb_isomap_reduce_{new_embed_len}.json'
    print(f'Dumping json to {output_f}...')
    with open(output_f, 'w') as json_out:
        json.dump(res.tolist(), json_out)
    print('Success!')

def autoencoder_dim_reduce():
    '''https://github.com/asdspal/dimRed/blob/master/autoencoder.ipynb'''
    print('Loading word vectors...')
    w_embed = util.torch_from_json('./data/word_emb.json')
    new_embed_len = w_embed.shape[1] // 2
    nn.Sequential(
        nn.Linear(w_embed.shape[1], new_embed_len, bias=False),
        nn.ReLU(),
        nn.Linear(new_embed_len, w_embed.shape[1], bias=False),
        nn.Sigmoid(),
    )
    #TODO train autoencoder

def kernel_pca_dim_reduce():
    '''Reduce the word vector dimensions by half using kernel PCA'''
    print('Loading word vectors...')
    w_embed = util.torch_from_json('./data/word_emb.json')
    new_len = w_embed.shape[1] // 2
    print(f'Performing Kernel PCA to reduce dimensions from {w_embed.shape[1]} to {new_len}...')
    pca = KernelPCA(n_components=new_len)
    res = pca.fit_transform(w_embed)
    output_f = f'./data/word_emb_kernel_pca_reduce_{new_len}.json'
    print(f'Dumping json to {output_f}...')
    with open(output_f, 'w') as json_out:
        json.dump(res.tolist(), json_out)
    print('Success!')

def main():
    '''Main'''
    is_dim_func = lambda v: (
        v in globals() and isinstance(globals()[v], types.FunctionType) and v.endswith('dim_reduce')
    )
    if len(sys.argv) <= 1 or not is_dim_func(sys.argv[1]):
        print(
            f'''Use one of the following as an argument: {
                [k for k in globals() if is_dim_func(k)]
            }''',
            file=sys.stderr)
        return
    globals()[sys.argv[1]]()

if __name__ == '__main__':
    main()
