# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
import numpy as np
from fairseq import utils
from fairseq.data import (
    Dictionary,
    FairseqDataset,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task

from torch.utils.data.dataloader import default_collate

logger = logging.getLogger(__name__)


@register_task("synthetic_task")
class SyntheticTask(LegacyFairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--sample-break-mode",
            default="complete",
            choices=["none", "complete", "complete_doc", "eos"],
            help='If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.',
        )
        parser.add_argument(
            "--tokens-per-sample",
            default=512,
            type=int,
            help="max number of total tokens over all segments "
            "per sample for BERT dataset",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.0,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.1,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.1,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--freq-weighted-replacement",
            default=False,
            action="store_true",
            help="sample random replacement words based on word frequencies",
        )
        parser.add_argument(
            "--mask-whole-words",
            default=False,
            action="store_true",
            help="mask whole words; you may also want to set --bpe",
        )
        parser.add_argument(
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )
        parser.add_argument(
            "--shorten-data-split-list",
            default="",
            help="comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',
        )

        parser.add_argument(
            "--vocab-size",
            type=int,
            default=1000,
            help="the size of the vocabulary"
        )

        parser.add_argument(
            "--max-seq-len",
            type=int,
            default=512,
        )

        parser.add_argument(
            "--min-seq-len",
            type=int,
            default=5,
        )

        parser.add_argument(
            "--dataset-train-size",
            type=int,
            default=512000,
        )
        parser.add_argument(
            "--dataset-valid-size",
            type=int,
            default=5000,
        )
        parser.add_argument(
            "--task-type",
            type=str,
            default='f',
            choices=['f', 'r'],
            help="type of synthetic tasks: 1) output first symbol at every position; 2) reverse the sequence"
        )

    def __init__(self, args):
        super().__init__(args)
        self.seed = args.seed
        self.vocab_size = args.vocab_size
        self.max_seq_len = args.max_seq_len
        self.min_seq_len = args.min_seq_len
        self.dataset_train_size = args.dataset_train_size
        self.dataset_valid_size = args.dataset_valid_size
        self.task_type = args.task_type

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        assert split in ['train', 'valid'], "invalid split: {}!".format(split)
        print(" Preparing data.")

        dataset = SimDataset(seed=self.seed, data_size=self.dataset_train_size if split == 'train' else self.dataset_valid_size,
                             vocab_size=self.vocab_size, max_seq_len=self.max_seq_len, min_seq_len=self.min_seq_len, task_type=self.task_type)

        logger.info('| Loaded {} with {} samples'.format(split, len(dataset)))
        self.datasets[split] = dataset

        return self.datasets[split]

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = RightPadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            pad_idx=self.source_dictionary.pad(),
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return SimDictionary(length=11+self.vocab_size)

    @property
    def target_dictionary(self):
        return SimDictionary(length=11+self.vocab_size)


class SimDataset(FairseqDataset):
    def __init__(self,
                 seed=1,
                 data_size=200000,
                 vocab_size=10000,
                 max_seq_len=512,
                 min_seq_len=5,
                 task_type='f'):

        self.data_size = data_size
        self.vocab_size = vocab_size # vocab index: 11 ~ vocab_size + 11
        self.shift_magic_number = 11
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.task_type = task_type
        assert self.task_type in ['f', 'r']
        self.seed(seed)

    def __getitem__(self, index):
        data = self._generate_data()
        return data

    def __len__(self):
        return self.data_size

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.max_seq_len

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.max_seq_len

    def _generate_data(self):
        # 1) length of the sequence
        sample_length = self.max_seq_len

        # 2) input sequence
        input_sample = np.random.choice(self.vocab_size - 20, sample_length, replace=True,) + self.shift_magic_number

        # 3) output sequence
        if self.task_type == 'f':

            # task: ETP
            output_sample = np.zeros_like(input_sample) + self.vocab_size - 2
            even_index = np.arange(sample_length // 2) * 2
            odd_index = even_index + 1
            output_sample[:sample_length // 2] = input_sample[odd_index]

        elif self.task_type == 'r':
            # task: PI
            output_sample =  1 + np.arange(128)
        else:
            raise NotImplementedError

        return {'x': torch.from_numpy(input_sample).long(),
                'y': torch.from_numpy(output_sample).long()}

    def seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def collater(self, samples):
        return default_collate(samples)

class SimDictionary():
    def __init__(self, padding_idx=1, length=10011):

        self.padding_idx = padding_idx
        self.length = length 
    
    def __len__(self):
        return self.length

    def pad(self):
        return self.padding_idx