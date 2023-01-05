import os

words_separator = ' '
punctuation_tags = ['#', '$', 'SYM', "''", ',', '-LRB-', '-RRB-', '.', ':', '``']
dataset_name = 'dependency_treebank'

model_type = 'glove'
embedding_dimension = 50
embedding_model_file = os.path.join('embeddings', f'{model_type}_{embedding_dimension}.kv')