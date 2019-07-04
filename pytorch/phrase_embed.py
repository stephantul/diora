"""
A script to embed every phrase in a dataset as a dense vector, then
to find the top-k neighbors of each phrase according to cosine
similarity.

1. Install missing dependencies.

    # More details: https://github.com/facebookresearch/faiss/blob/master/INSTALL.md
    conda install faiss-cpu -c pytorch

2. Prepare data. For example, the chunking dataset from CoNLL 2000.

    wget https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz
    gunzip train.txt.gz
    python diora/misc/convert_conll_to_jsonl.py --path train.txt > conll-train.jsonl

3. Run this script.

    python diora/scripts/phrase_embed.py \
        --batch_size 10 \
        --emb w2v \
        --embeddings_path ~/data/glove.6B/glove.6B.50d.txt \
        --hidden_dim 50 \
        --log_every_batch 100 \
        --save_after 1000 \
        --data_type conll_jsonl \
        --validation_path ./conll-train.jsonl \
        --validation_filter_length 10

Can control the number of neighbors to show with the `--k_top` flag.

Can control the number of candidates to consider with `--k_candidates` flag.

"""
import itertools
import torch
import numpy as np

from train import argument_parser, parse_args, configure
from train import get_validation_dataset, get_validation_iterator
from train import build_net

from diora.logging.configuration import get_logger
from reach import Reach


def get_cell_index(entity_labels, i_label=0, i_pos=1, i_size=2):
    def helper():
        for i, lst in enumerate(entity_labels):
            for el in lst:
                if el is None:
                    continue
                pos = el[i_pos]
                size = el[i_size]
                label = el[i_label]
                yield (i, pos, size, label)
    lst = list(helper())
    if len(lst) == 0:
        return None, []
    batch_index = [x[0] for x in lst]
    positions = [x[1] for x in lst]
    sizes = [x[2] for x in lst]
    labels = [x[3] for x in lst]

    return batch_index, positions, sizes, labels


def get_many_cells(diora, chart, batch_index, positions, sizes):
    cells = []
    length = diora.length

    idx = []
    for bi, pos, size in zip(batch_index, positions, sizes):
        level = size - 1
        offset = diora.index.get_offset(length)[level]
        absolute_pos = offset + pos
        idx.append(absolute_pos)

    cells = chart[batch_index, idx]

    return cells


def get_many_phrases(batch, batch_index, positions, sizes):
    batch = batch.tolist()
    lst = []
    for bi, pos, size in zip(batch_index, positions, sizes):
        phrase = tuple(batch[bi][pos:pos+size])
        lst.append(phrase)
    return lst


class BatchRecorder(object):
    def __init__(self, dtype={}):
        super(BatchRecorder, self).__init__()
        self.cache = {}
        self.dtype = dtype
        self.dtype2flatten = {
            'list': self._flatten_list,
            'np': self._flatten_np,
            'torch': self._flatten_torch,
        }

    def _flatten_list(self, v):
        return list(itertools.chain(*v))

    def _flatten_np(self, v):
        return np.concatenate(v, axis=0)

    def _flatten_torch(self, v):
        return torch.cat(v, 0).cpu().data.numpy()

    def get_flattened_result(self):
        def helper():
            for k, v in self.cache.items():
                flatten = self.dtype2flatten[self.dtype.get(k, 'list')]
                yield k, flatten(v)
        return {k: v for k, v in helper()}

    def record(self, **kwargs):
        for k, v in kwargs.items():
            self.cache.setdefault(k, []).append(v)


def run(options):
    logger = get_logger()

    validation_dataset = get_validation_dataset(options)
    validation_iterator = get_validation_iterator(options, validation_dataset)
    word2idx = validation_dataset['word2idx']
    embeddings = validation_dataset['embeddings']

    idx2word = {v: k for k, v in word2idx.items()}

    logger.info('Initializing model.')
    trainer = build_net(options, embeddings, validation_iterator)
    diora = trainer.net.diora

    # 1. Get all relevant phrase vectors.
    dtype = {
        'example_ids': 'list',
        'labels': 'list',
        'positions': 'list',
        'sizes': 'list',
        'phrases': 'list',
        'inside': 'torch',
        'outside': 'torch',
    }
    batch_recorder = BatchRecorder(dtype=dtype)
    # Eval mode.
    trainer.net.eval()

    batches = validation_iterator.get_iterator(random_seed=options.seed)

    logger.info('Beginning to embed phrases.')

    strings = []
    with torch.no_grad():
        for i, batch_map in enumerate(batches):
            sentences = batch_map['sentences']
            length = sentences.shape[1]

            # Skips very short examples.
            if length <= 2:
                continue
            strings.extend(["".join([idx2word[idx] for idx in x])
                           for x in sentences.numpy()])
            trainer.step(batch_map, train=False, compute_loss=False)

            batch_result = {}
            batch_result['inside'] = diora.inside_h[:, -1]
            batch_result['outside'] = diora.outside_h[:, -1]
            batch_recorder.record(**batch_result)

    result = batch_recorder.get_flattened_result()

    # 2. Build an index of nearest neighbors.
    vectors = np.concatenate([result['inside'], result['outside']], axis=1)
    print(len(strings), vectors.shape)
    r = Reach(vectors, strings)

    for s in strings:
        print(s)
        print(r.most_similar(s))


if __name__ == '__main__':
    parser = argument_parser()
    parser.add_argument('--k_candidates', default=100, type=int)
    parser.add_argument('--k_top', default=3, type=int)
    options = parse_args(parser)
    configure(options)

    run(options)
