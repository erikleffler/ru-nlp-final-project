import re
import string

import tensorflow as tf


class TextMetric(tf.keras.metrics.Metric):
    def __init__(self, tokenizer, name="text_metric", **kwargs):
        super(TextMetric, self).__init__(name=name, **kwargs)

    # Taken from the MRQA shared task 2019
    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def text_from_tokens(self, token_ids):
        return self.tokenizer.decode(token_ids)

