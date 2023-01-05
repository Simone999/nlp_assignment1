import argparse
import functools
import pickle
import sys
from typing import Dict, Literal, Optional, Tuple
import gensim
import gensim.downloader as gloader
import numpy as np

from enum import Flag, auto
import pandas as pd

from tqdm import tqdm
from preprocessing import preprocess_text
from util import *


def download_embedding_model(model_type: str = 'glove', embedding_dimension: int = 50) -> gensim.models.keyedvectors.KeyedVectors:
    """
    Loads a pre-trained word embedding model via gensim library.

    :param model_type: name of the word embedding model to load.
    :param embedding_dimension: size of the embedding space to consider

    :return
        - pre-trained word embedding model (gensim KeyedVectors object)
    """
    download_path = ""
    if model_type.strip().lower() == 'word2vec':
        download_path = "word2vec-google-news-300"

    elif model_type.strip().lower() == 'glove':
        download_path = "glove-wiki-gigaword-{}".format(embedding_dimension)
    elif model_type.strip().lower() == 'fasttext':
        download_path = "fasttext-wiki-news-subwords-300"
    else:
        raise AttributeError(
            "Unsupported embedding model type! Available ones: word2vec, glove, fasttext")

    try:
        emb_model = gloader.load(download_path)
    except ValueError as e:
        print("Invalid embedding model name! Check the embedding dimension:")
        print("Word2Vec: 300")
        print("Glove: 50, 100, 200, 300")
        print('FastText: 300')
        raise e

    return emb_model


class OOVStrategy(Flag):
    NONE = 0
    COMPOSED_WORDS = auto()
    NUMBER_EMBEDDING = auto()
    CAPITALIZED_EMBEDDING = auto()


class EmbeddingMatrix():
    def __init__(self,
                 embedding_model: gensim.models.keyedvectors.KeyedVectors,
                 oov_strategies: OOVStrategy = OOVStrategy.NONE,
                 padding=True,
                 number_token='[NUMBER]',
                 capitalize_vector: np.ndarray = None) -> None:
        """
        Builds the embedding matrix of a specific dataset given a pre-trained word embedding model

        :param embedding_model: pre-trained word embedding model (gensim wrapper)
        :param embedding_dimension: dimension of the vectors in the embedding space
        :param word_to_idx: vocabulary map (word -> index) (dict)
        """
        self.__composed_words_delimiters = re.compile(r'\b' + one_of_as_regex(['-', '_', '/']) + r'\b')
        self.oov_strategies = oov_strategies
        self.number_token = number_token

        tokens, vectors = self.__get_embeddings(embedding_model, padding)
        self.embedding_model = gensim.models.keyedvectors.KeyedVectors(
            embedding_model.vector_size)
        self.embedding_model.add_vectors(tokens, vectors)

        self.capitalize_vector = capitalize_vector
        if OOVStrategy.CAPITALIZED_EMBEDDING in self.oov_strategies and capitalize_vector is None:
            self.capitalize_vector = np.repeat(
                0.1, embedding_model.vector_size)

    def __get_embeddings(self, embedding_model: gensim.models.keyedvectors.KeyedVectors, padding: bool):
        tokens = []
        vectors = []
        embedding_word_to_id = embedding_model.key_to_index.copy()

        if padding:
            tokens.append('')
            vectors.append(np.zeros(embedding_model.vector_size))

        if OOVStrategy.NUMBER_EMBEDDING in self.oov_strategies:
            embedding_numbers = [
                word for word in embedding_word_to_id.keys() if is_number(word)]
            number_vector = embedding_model.get_mean_vector(embedding_numbers)
            remove_items(embedding_word_to_id, embedding_numbers)
            tokens.append(self.number_token)
            vectors.append(number_vector)

        for token, id in tqdm(embedding_word_to_id.items()):
            tokens.append(token)
            vectors.append(embedding_model[id])

        return tokens, vectors

    def add_tokens(self, tokens: Iterable[str]):
        tokens = set(tokens)
        previous_terms = set(self.embedding_model.key_to_index.keys())
        oov = list(tokens.difference(previous_terms))
        n_oov = len(oov)
        print('Number of OOV terms:', n_oov)

        self.embedding_model.add_vectors(oov, np.zeros(
            (n_oov, self.embedding_model.vector_size)))
        for token in tqdm(oov):
            self.add_token(token)

        new_tokens = list(
            set(self.embedding_model.key_to_index.keys()).difference(previous_terms))
        return new_tokens

    def add_token(self, token: str):
        # If already created, return
        if np.any(self.embedding_model[self.embedding_model.key_to_index[token]] != 0):
            return

        self.embedding_model[token] = np.random.uniform(
            low=-0.05, high=0.05, size=self.embedding_model.vector_size)

        if OOVStrategy.CAPITALIZED_EMBEDDING in self.oov_strategies and starts_with_uppercase(token):
            # we ensure that the lower case form of the word is in the embedding matrix
            self.embedding_model[token] = self.__get_vector_or_handle_as_oov(
                token.lower()) + self.capitalize_vector

        elif OOVStrategy.COMPOSED_WORDS in self.oov_strategies:
            sub_tokens = self.__split_composed(token)
            if len(sub_tokens) > 1:
                sub_tokens_vectors = [self.__get_vector_or_handle_as_oov(
                    sub_token) for sub_token in sub_tokens]
                self.embedding_model[token] = np.mean(
                    sub_tokens_vectors, axis=0)

    def __get_vector_or_handle_as_oov(self, token: str):
        if OOVStrategy.NUMBER_EMBEDDING in self.oov_strategies and is_number(token):
            return self.embedding_model.get_vector(self.number_token)

        # if not present, initialize it
        if not self.embedding_model.has_index_for(token):
            self.embedding_model.add_vector(
                token, np.zeros(self.embedding_model.vector_size))

        id = self.embedding_model.key_to_index[token]
        vector = self.embedding_model[id]
        # if not created, create it
        if np.all(vector == 0):
            self.add_token(token)
            vector = self.embedding_model[id]

        return vector

    def __split_composed(self, token: str) -> List[str]:
        return re.split(self.__composed_words_delimiters, token)

    def __is_composed(self, token: str) -> bool:
        return len(self.__split_composed(token)) > 1

    def vectors(self) -> np.ndarray:
        return self.embedding_model.vectors

    def vocabulary(self) -> Dict[str, int]:
        return self.embedding_model.key_to_index

    def store(self, filename: str):
        with open(filename, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename: str):
        with open(filename, 'rb') as file:
            return pickle.load(file)

def add_dataset_terms(embedding_matrix: EmbeddingMatrix, dataset, dataset_name: str, words_separator=' '):
    print(f'Adding {dataset_name} terms ...')
    tokens = flatten(dataset.sentence.str.split(words_separator))
    new_tokens = embedding_matrix.add_tokens(tokens)
    print('Number of new tokens:', len(new_tokens))
    print('New tokens:', new_tokens)
    print('-'*30)

def get_oov_strategies(text_strategy, number_strategy, composed_words):
    oov_strategies = OOVStrategy.NONE
    if composed_words:
        oov_strategies |= OOVStrategy.COMPOSED_WORDS
    if text_strategy == 'lower_up':
        oov_strategies |= OOVStrategy.CAPITALIZED_EMBEDDING
    if number_strategy == 'token':
        oov_strategies |= OOVStrategy.NUMBER_EMBEDDING

    return oov_strategies


def preprocess_and_create_embedding(
    dataset: pd.DataFrame,
    embedding_model_file: str,
    text_strategy: Optional[Literal['lower', 'lower_up']] = None,
    number_strategy: Optional[Literal['token']] = None,
    composed_words=False,
    words_separator=' '
) -> Tuple[pd.DataFrame, EmbeddingMatrix]:

    dataset = dataset.copy()
    preprocess = functools.partial(
        preprocess_text, text_strategy=text_strategy, number_strategy=number_strategy, words_separator=words_separator)
    dataset['sentence'] = dataset['sentence'].apply(preprocess)
    train_data, val_data, test_data = split_dataset(dataset)

    oov_strategies = get_oov_strategies(
        text_strategy=text_strategy, number_strategy=number_strategy, composed_words=composed_words)
    embedding_model = gensim.models.keyedvectors.KeyedVectors.load(embedding_model_file)
    embedding_matrix = EmbeddingMatrix(embedding_model, oov_strategies=oov_strategies)
    del embedding_model

    add_dataset_terms(embedding_matrix, train_data, 'training set')
    add_dataset_terms(embedding_matrix, val_data, 'validation set')
    add_dataset_terms(embedding_matrix, test_data, 'test set')

    return dataset, embedding_matrix

def preprocess_and_load_embedding(
  dataset:pd.DataFrame,
  text_strategy:Optional[Literal['lower', 'lower_up']]=None,
  number_strategy:Optional[Literal['token']]=None,
  composed_words=False
) -> Tuple[pd.DataFrame, EmbeddingMatrix]:
    dataset = dataset.copy()
    preprocess = functools.partial(preprocess_text, text_strategy=text_strategy, number_strategy=number_strategy)
    dataset['sentence'] = dataset['sentence'].apply(preprocess)
    
    filename = os.path.join('embeddings', generate_filename(text_strategy, number_strategy, composed_words))
    embedding_matrix = EmbeddingMatrix.load(filename)

    return dataset, embedding_matrix

def generate_filename(
    text_strategy: Optional[Literal['lower', 'lower_up']],
    number_strategy: Optional[Literal['token']],
    composed_words: bool
):
    filename = 'embedding_matrix'
    if text_strategy:
        filename += f't_{text_strategy}'
    if number_strategy:
        filename += f'n_{number_strategy}'        
    if composed_words:
        filename += '_composed'
    return filename + '.pkl'

embedding_strategies={
  'text_strategy': [None, 'lower', 'lower_up'],
  'number_strategy': [None, 'token'],
  'composed_words': [False, True],
}

def parse_args():
    parser = argparse.ArgumentParser('Embedding generator')
    parser.add_argument('dataset', help='Input data pickle file.')
    parser.add_argument('embedding_model', help='Gensim model file')
    
    parser.add_argument('--all', dest='generate_all', action='store_true')

    parser.add_argument('--text-strategy', dest="text_strategy",
                        choices=['lower', 'lower_up'])
    parser.add_argument('--number-strategy',
                        dest="number_strategy", choices=['token'])
    parser.add_argument('--composed-words',
                        dest="composed_words", action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main():
    dataset = pd.read_pickle(args.dataset)
    embedding_model_file = args.embedding_model
    if args.generate_all:
        strategies = generate_combinations(embedding_strategies)
    else:
        strategy = {
            'text_strategy': args.text_strategy,
            'number_strategy': args.number_strategy,
            'composed_words': args.composed_words
        }        
        strategies = [strategy]
    
    for strategy in strategies:
        print(strategy)
        _, embedding_matrix = preprocess_and_create_embedding(dataset, embedding_model_file, **strategy)
        print('='*30)
        filename = os.path.join('embeddings', generate_filename(**strategy))
        embedding_matrix.store(filename)
        del embedding_matrix

if __name__ == '__main__':
    args = parse_args()
    main()
