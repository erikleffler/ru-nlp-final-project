# ru-nlp-final-project
The final project in the NLP course at reykjavik university. 

The model code is split into two directories, `code/albert-supervised-finetune` and `code/albert-pseudo-labeling`. 
These directories represent two different types of google cloud ai-platform jobs where the main code is stored under the `trainer` subdirectory.
The entry point for a google cloud ai-platform job is the file `trainer/task.py` file. For convenience, there are shell scripts in both directories
that can be used to send out a remot training job, or to train locally for debugging purposes. To run these you'd need install and configure the
[google cloud sdk](https://cloud.google.com/sdk/docs/install) (and to run a remot job you need a google cloud account with ai-platform enabled). 

I've included the test predictions under the `code/evaluate` subdirectory. To get the test results form when `alpha = 0.32` for example, from the 
`code/evaluate` subdirectory run `python mrqa_official_eval.py ../../data/BioASQ.jsonl pseudo0.16/jobs_questionable_albert20201111_104819_test_preds.jsonl --skip-no-answer`.

I dumped my conda environment into `conda_dump_requirements.txt`.
