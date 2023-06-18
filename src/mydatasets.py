import os
import copy
import json
import random
from tqdm import tqdm
from typing import Callable, Any

from datasets import load_dataset
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

from log import print
from prompts import QuestionPart, Exemplar, idx_to_ltr

IGNORE_INDEX = -100
REPRODUCIBILITY_SEED = 0


class MyDataset(Dataset):
    def __init__(self, data_args, tokenizer, dataset_info, split):
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.split = split
        self.sample_size = dataset_info.sample_size
        self.prompt_type = dataset_info.prompt_type

        save_dir = os.path.join(data_args.data_dir, data_args.dataset_name, data_args.data_tag)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        save_file = os.path.join(save_dir, f'{split}.pt')
        if data_args.refresh or not os.path.exists(save_file):
            dataset = load_dataset(dataset_info.path, name=dataset_info.name, split=split)
            self.data = self.process(dataset_info.extractor, dataset, save_file)
        else:
            print('Loading data from', save_file)
            self.data = torch.load(save_file)
        print('Data size:', len(self.data))
        print('Data format:', self.data[0])
        print('Max length:', max([len(d['input_ids']) for d in self.data])) if self.split == 'train' else \
            print('Max length:', max([max([len(d) for d in dd['input_ids']]) for dd in self.data]))

    def process(self, extractor, dataset, save_file):
        data = []
        for instance in tqdm(dataset):
            exemplar = Exemplar(**extractor(instance))
            if self.prompt_type == 'brown':
                prompt = exemplar.get_brown_prompt()
            else:
                prompt = exemplar.get_natural_prompt()
            source = prompt['source']

            targets = []

            def _tokenize_fn(source, target):
                targets.append(target)
                example = f"{source}{target}"
                example_tokenized = self.tokenizer.encode(example, truncation=True, max_length=self.data_args.data_max_length)
                example_tokenized = example_tokenized + [self.tokenizer.eos_token_id]
                source_tokenized = self.tokenizer.encode(source)

                input_ids = example_tokenized
                labels = copy.deepcopy(input_ids)
                if not self.data_args.train_on_inputs:
                    labels = np.array(labels)
                    labels[:len(source_tokenized) - 1] = IGNORE_INDEX
                return input_ids, labels

            if self.split == 'train':
                input_ids, labels = _tokenize_fn(source, prompt['target'])
            else:
                input_ids = []
                labels = []
                for choice in prompt['choices']:
                    op_input_ids, op_labels = _tokenize_fn(source, choice)
                    input_ids.append(op_input_ids)
                    labels.append(op_labels)

            data.append({'input_ids': input_ids,
                         'labels': labels,
                         'source': source,
                         'target': targets,
                         'answer': exemplar.answer_idx})

        if self.sample_size > 0 and len(data) > self.sample_size:
            random.seed(REPRODUCIBILITY_SEED)
            possible_idxs = list(range(len(data)))
            sampled_idxs = random.sample(possible_idxs, self.sample_size)
            data = [data[i] for i in sampled_idxs]
            print(f'Sampled {self.sample_size} examples from {len(possible_idxs)} examples.')

        torch.save(data, save_file)
        print('Saving data to', save_file)
        return data

    def concat_exemplars(self, exemplars):
        exemplar_prompts = [f"{e['source']}{e['target'][0]}" for e in exemplars]
        exemplars = "\n\n".join(exemplar_prompts)
        return exemplars

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'],
            'labels': self.data[idx]['labels']
        }


@dataclass
class DatasetInfo:
    path: str = None
    exemplar_split: str = None
    eval_split: str = None
    test_split: str = None
    extractor: Callable = Any
    name: str = None
    data_dir: str = None
    sample_size: int = -1
    prompt_type: str = 'brown'


def get_dataset_info(dataset_name):
    if dataset_name == 'boolq':
        return DatasetInfo(
            path="super_glue",
            name="boolq",
            exemplar_split="train",
            eval_split="validation",
            sample_size=1000,
            extractor=lambda row: {
                "parts": [
                    QuestionPart(
                        f"{row['passage']} {row['question']}",
                    ),
                ],
                "choices": [
                    'No', 'Yes'
                ],
                "answer_idx": int(row["label"])
            }
        )
    # elif dataset_name == 'cb':
    #     return DatasetInfo(
    #         path="super_glue",
    #         name="cb",
    #         exemplar_split="train",
    #         eval_split="validation",
    #         sample_size=1000,
    #         extractor=lambda row: {
    #             "parts": [
    #                 QuestionPart(
    #                     f"Suppose {row['premise']} Can we infer that \"{row['hypothesis']}\"? Yes, No, or Maybe?",
    #                 ),
    #             ],
    #             "choices": [
    #                 'Yes', 'No', 'Maybe'
    #             ],
    #             "answer_idx": int(row["label"])
    #         }
    #     )
    elif dataset_name == 'multirc':
        return DatasetInfo(
            path="super_glue",
            name="multirc",
            exemplar_split="train",
            eval_split="validation",
            sample_size=1000,
            extractor=lambda row: {
                "parts": [
                    QuestionPart(
                        f"{row['paragraph']}",
                    ),
                    QuestionPart(
                        f"{row['question']}",
                        tag='Question'
                    ),
                    QuestionPart(
                        f'I found this answer "{row["answer"]}". Is that correct? Yes or No?',
                    ),
                ],
                "choices": [
                    'No', 'Yes'
                ],
                "answer_idx": int(row["label"])
            }
        )
    elif dataset_name == 'rte':
        return DatasetInfo(
            path="super_glue",
            name="rte",
            exemplar_split="train",
            eval_split="validation",
            sample_size=1000,
            extractor=lambda row: {
                "parts": [
                    QuestionPart(
                        f"{row['premise']}\nDoes this mean that \"{row['hypothesis']}\" is true? Yes or No?",
                    ),
                ],
                "choices": [
                    'Yes', 'No'
                ],
                "answer_idx": int(row["label"])
            }
        )
    elif dataset_name == 'wic':
        return DatasetInfo(
            path="super_glue",
            name="wic",
            exemplar_split="train",
            eval_split="validation",
            sample_size=1000,
            extractor=lambda row: {
                "parts": [
                    QuestionPart(
                        f"Does the word \"{row['word']}\" have the same meaning in these two sentences? Yes, No?\n{row['sentence1']}\n{row['sentence2']}",
                    ),
                ],
                "choices": [
                    'No', 'Yes'
                ],
                "answer_idx": int(row["label"])
            }
        )
    elif dataset_name == 'wsc':
        return DatasetInfo(
            path="super_glue",
            name="wsc",
            exemplar_split="train",
            eval_split="validation",
            sample_size=1000,
            extractor=lambda row: {
                "parts": [
                    QuestionPart(
                        f"{row['text']}\nIn the previous sentence, does the pronuon \"{row['span2_text']}\" refer to \"{row['span1_text']}\"? Yes or No?",
                    ),
                ],
                "choices": [
                    'No', 'Yes'
                ],
                "answer_idx": int(row["label"])
            }
        )
    elif dataset_name == 'copa':
        return DatasetInfo(
            path="super_glue",
            name="copa",
            exemplar_split="train",
            eval_split="validation",
            sample_size=1000,
            prompt_type='natural',
            extractor=lambda row: {
                "parts": [
                    QuestionPart(
                        f"{row['premise']} so " if row['question'] == 'effect' else f"{row['premise']} because ",
                    ),
                ],
                "choices": [
                    row['choice1'], row['choice2']
                ],
                "answer_idx": int(row["label"])
            }
        )
    elif dataset_name == 'record':
        return DatasetInfo(
            path="super_glue",
            name="record",
            exemplar_split="train",
            eval_split="validation",
            sample_size=1000,
            extractor=process_record
        )
    else:
        raise NotImplementedError


def process_record(row):
    def record_clean_choices(row):
        if len(row['answers']) == 1:
            return row['entities'], row['entities'].index(row['answers'][0])

        new_entities = []
        for entity in row['entities']:
            if entity in row['answers'][1:]:
                continue
            new_entities.append(entity)
        return new_entities, new_entities.index(row['answers'][0])

    choices, answer_idx = record_clean_choices(row)
    return {
                "parts": [
                    QuestionPart(
                        "{}\n{}\nQuestion: What is the \"@placeholder\"?".format(row['passage'].replace('@highlight\n', '- '), row['query']),
                    ),
                ],
                "choices": choices,
                "answer_idx": answer_idx
            }


if __name__ == '__main__':
    from transformers import HfArgumentParser
    from arguments import ModelArguments, DataArguments
    from transformers import AutoTokenizer

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    model_args.model_name_or_path = '/home/klv/llama_hf/7B'
    data_args.dataset_name = 'record'
    data_args.refresh = True
    data_args.data_tag = 'debug'
    train_on_inputs = False
    data_args.data_max_length = 512

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        padding_side='left'
    )
    tokenizer.pad_token_id = 0

    dataset_info = get_dataset_info(data_args.dataset_name)
    train_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.exemplar_split)
    eval_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.eval_split)
    # test_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.test_split)



