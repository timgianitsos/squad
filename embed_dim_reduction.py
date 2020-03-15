'''
Perform dimensionality reduction on the word vectors
'''

import sys
import types

from sklearn.decomposition import PCA
import ujson as json #pylint:disable=import-error

import util

def pca_dim_reduce():
    '''Reduce the word vector dimensions by half using PCA'''
    print('Loading word vectors...')
    word_embed = util.torch_from_json('./data/word_emb.json')
    new_embed_len = word_embed.shape[1] // 2
    print(f'Performing PCA to reduce dimensions from {word_embed.shape[1]} to {new_embed_len}...')
    pca = PCA(n_components=new_embed_len)
    res = pca.fit_transform(word_embed)
    output_f = f'./data/word_emb_pca_reduce_{new_embed_len}.json'
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
