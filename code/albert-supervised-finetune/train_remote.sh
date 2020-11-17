#!/bin/bash

set -e

eval=1 # default

# Read cmdline opts

positional=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
	--config)
		config="$2"
		shift # past argument
		shift # past value

		case $config in
			tpu)
				tpu=1
				distributed=1
				config='configs/config_tpu.yaml'
				region="europe-west4" # Has TPU's
				batch_size=32
				;;
			gpu-basic)
				tpu=0
				distributed=0
				config='configs/config_gpu_basic.yaml'
				region="europe-west1" # Has K80s
				batch_size=8
				;;
			gpu-V100-single)
				tpu=0
				distributed=0
				config='configs/config_gpu_V100_single.yaml'
				region="europe-west4" # Has V100's
				batch_size=16
				;;
			gpu-V100-distributed)
				tpu=0
				distributed=1
				config='configs/config_gpu_V100_distributed.yaml'
				region="europe-west4" # Has V100's
				batch_size=16
				;;
			*)
				echo "Invalid config parameter, --config must be one of: tpu | gpu-basic | gpu-V100-single | gpu-V100-distributed"
				;;
		esac
		;;
	--no-eval)
		eval=0
		shift
		echo "Not doing kfolds evaluation"
		;;


	*)    # unknown option
		positional+=("$1") # save it in an array for later
		shift # past argument
		;;
esac
done
set -- "${positional[@]}" # restore positional parameters

now=$(date +"%Y%m%d_%H%M%S")

job_name=questionable_albert${now}
bucket_name="ru-nlp-final-project"
job_dir="gs://${bucket_name}/jobs/${job_name}"
data_path="data/BioASQ-train.jsonl"
test_data_path="data/BioASQ-test.jsonl"


gcloud ai-platform jobs submit training ${job_name} \
	--package-path trainer/ \
	--module-name trainer.task \
	--region $region \
	--job-dir $job_dir \
	--config $config \
	--runtime-version 2.2 \
	--stream-logs \
	-- \
	--bucket-name $bucket_name \
	--data-path $data_path \
	--test-data-path $test_data_path \
	--num-epochs 15 \
	--max-length 512 \
	--batch-size $batch_size \
	--learning-rate 3e-05 \
	--training-steps 500000 \
	--warmup-steps 50000 \
	--eval-steps 500 \
	--kfold-eval $eval \
	--num-folds  5 \
	--pre-train-mlm-head-steps 50000 \
	--tpu $tpu \
	--distributed $distributed \
	--xla 1 \
	--amp 1 \
	--checkpoint-freq 50000 \
	--remote 1
