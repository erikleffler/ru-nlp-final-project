#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")

job_name=questionable_albert${now}
bucket_name="ru-nlp-final-project"
bio_data_path="../../data/BioASQ-train.jsonl"
squad_data_path="../../data/SQuAD-shuffled.jsonl"
test_data_path="../../data/BioASQ-test.jsonl"
job_dir="/Users/erikleffler1/programming/ru-courses/natural-language-processing/ru-nlp-final-project/code/albert-supervised-finetune/local-job-dir"

gcloud ai-platform local train \
	--package-path trainer/ \
	--module-name trainer.task \
	--job-dir $job_dir \
	-- \
	--bucket-name $bucket_name \
	--bert-dir $bert_dir \
	--squad-data-path $squad_data_path \
	--bio-data-path $bio_data_path \
	--test-data-path $test_data_path \
	--num-epochs 1 \
	--max-length 4 \
	--batch-size 2 \
	--learning-rate 3e-09 \
	--training-steps 6 \
	--pre-train-mlm-head-steps 5 \
	--warmup-steps 5 \
	--eval-steps 5 \
	--pseudo-weight '0.05' \
	--num-folds  2 \
	--xla 0 \
	--amp 0 \
	--checkpoint-freq 25 \
	--remote 0
