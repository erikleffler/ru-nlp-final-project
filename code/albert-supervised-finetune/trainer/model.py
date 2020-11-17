import tensorflow as tf
from transformers import TFAlbertForQuestionAnswering


class AlbertQA(tf.keras.model):
    def __init__(self, model_string, name="albert_for_qa", **kwargs):
        super(AlbertQA, self).__init__(name=name, **kwargs)
        self.albert = TFAlbertForQuestionAnswering.from_pretrained(model_string)

