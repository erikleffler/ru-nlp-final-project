import re
import string
from collections import Counter

import tensorflow as tf


# Average of f1 over tokens for each batch
class MrqaTokenF1Mean(tf.keras.metrics.Metric):
    def __init__(self, name="mrqa_token_f1_mean", **kwargs):
        super(MrqaTokenF1Mean, self).__init__(name=name, **kwargs)

        self.f1 = 0
        self.total = 0

    def update_state(
        self, batch_ground_truths, batch_model_answers, sample_weight=None
    ):
        self.total += len(batch_model_answers)

        for ground_truths, model_answer in zip(
            batch_ground_truths, batch_model_answers
        ):

            # Get the answer from ground truths that is closest match
            scores_for_ground_truths = []
            for ground_truth in ground_truths:
                score = self.token_f1_score(
                    model_answer,
                    ground_truth,
                )
                scores_for_ground_truths.append(score)

            self.f1 += max(scores_for_ground_truths)

    def result(self):
        return float(self.f1) / self.total

    def reset_states(self):
        self.f1 = 0
        self.total = 0

    def token_f1_score(self, model_answer, ground_truth):
        prediction_tokens = normalize_answer(model_answer).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


class MrqaExactMatch(tf.keras.metrics.Metric):
    def __init__(self, name="mrqa_exact_match", **kwargs):
        super(MrqaExactMatch, self).__init__(name=name, **kwargs)

        self.matches = 0
        self.total = 0

    def update_state(
        self, batch_ground_truths, batch_answers, sample_weight=None
    ):
        self.total += len(batch_answers)

        for ground_truths, answer in zip(batch_ground_truths, batch_answers):
            if normalize_answer(answer) in map(
                normalize_answer, ground_truths
            ):
                self.matches += 1

    def result(self):
        return 100.0 * float(self.matches) / self.total

    def reset_states(self):
        self.matches = 0
        self.total = 0


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
