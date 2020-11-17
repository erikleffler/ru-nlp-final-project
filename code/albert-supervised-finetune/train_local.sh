#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")

job_name=questionable_albert${now}
bucket_name="ru-nlp-final-project"
data_path="../../data/BioASQ-train.jsonl"
test_data_path="../../data/BioASQ-test.jsonl"
job_dir="/Users/erikleffler1/programming/ru-courses/natural-language-processing/ru-nlp-final-project/code/bert-squad-regular/local-job-dir"

gcloud ai-platform local train \
	--package-path trainer/ \
	--module-name trainer.task \
	--job-dir $job_dir \
	-- \
	--bucket-name $bucket_name \
	--bert-dir $bert_dir \
	--data-path $data_path \
	--test-data-path $test_data_path \
	--num-epochs 1 \
	--max-length 4 \
	--batch-size 1 \
	--learning-rate 3e-05 \
	--training-steps 6 \
	--pre-train-mlm-head-steps 5 \
	--warmup-steps 5 \
	--eval-steps 5 \
	--num-folds  2 \
	--xla 0 \
	--amp 0 \
	--checkpoint-freq 25 \
	--remote 0
