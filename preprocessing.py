# TODO: Handle abbreviations like USA and U.S.A.
import re
from typing import Literal, Optional

def lower_if_upper(word: str):
  return word.lower() if str.isupper(word) else word
  
def preprocess_text(sentence: str, text_strategy:Optional[Literal['lower', 'lower_up']]=None, number_strategy:Optional[Literal['token']]=None, words_separator=' ') -> str:
  # restore slash (/)
  sentence = sentence.replace('\\/', '/')
  
  if text_strategy == 'lower':
    sentence = sentence.lower()
  elif text_strategy == 'lower_up':
    sentence = words_separator.join(lower_if_upper(word) for word in sentence.split(words_separator))

  if number_strategy == 'token':
    # a32.4b -> a32.4b, 32.4 -> [NUMBER]
    sentence = re.sub(r'(\s|^)([\-]?[0-9]*[\.,]?[0-9]+)(?=\s|$)', r'\1[NUMBER]', sentence)

  return sentence