#!/usr/bin/env bash
model_name_path="microsoft/deberta-large"
classifier="models/hf_classifier.py"
#train_file="/data/users/jfchen/fact-check-via-raw-evidence/data/webR-gnrtQ/gpt-sum/train-003-multi-sites-restricted-time-restricted-05-12.jsonl"
#validation_file="/data/users/jfchen/fact-check-via-raw-evidence/data/webR-gnrtQ/gpt-sum/dev-003-multi-sites-restricted-time-restricted-05-12.jsonl"
##test_file="./data/webR-gnrtQ/gpt-sum/table1/test-003-multi-sites-restricted-time-restricted-05-04.jsonl"
#test_file="/data/users/jfchen/fact-check-via-raw-evidence/data/webR-gnrtQ/gpt-sum/table1/test-003-multi-sites-restricted-time-restricted-05-04.jsonl"

#echo ${test_file}
#echo ${train_file}
#echo ${validation_file}
#exit

train_file="./data/train-site-restricted-wo-search-results.jsonl"
validation_file="./data/dev-site-restricted-wo-search-results.jsonl"
test_file="./data/test-site-restricted-wo-search-results.jsonl"
output_dir="./model_output/"
eval_steps=100
max_seq_length=512
train_batch_size=8
eval_batch_size=16
epochs=25
learning_rate=3e-5
gradient_accumulation_steps=2
use_claim=true
use_justification=false
use_decomposed_question=false
use_bm25_retrieval=false
use_answer=false
use_qa_pairs=false
use_gpt_rationale=true
metric_for_best_model="MAE"
gpu_num=1
topk_docs=1
topk_sents=10
seed=290032
num_class=6

while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi
  shift
done

python -m torch.distributed.launch --nproc_per_node ${gpu_num} --master_port 1235 ${classifier} \
--model_name_or_path ${model_name_path} \
--cache_dir "./cache" \
--train_file ${train_file} \
--validation_file ${validation_file} \
--test_file ${test_file} \
--do_train true \
--do_eval true \
--do_predict true \
--logging_strategy steps \
--logging_first_step true \
--logging_steps 100 \
--evaluation_strategy steps \
--max_seq_length ${max_seq_length} \
--eval_steps ${eval_steps} \
--save_strategy steps \
--save_steps ${eval_steps} \
--save_total_limit 2 \
--metric_for_best_model ${metric_for_best_model} \
--load_best_model_at_end true \
--greater_is_better false \
--gradient_accumulation_steps ${gradient_accumulation_steps} \
--num_train_epochs ${epochs} \
--learning_rate ${learning_rate} \
--output_dir ${output_dir} \
--overwrite_output_dir true \
--per_device_train_batch_size ${train_batch_size} \
--per_device_eval_batch_size ${eval_batch_size} \
--report_to none \
--seed ${seed} \
--use_justification ${use_justification} \
--dataset_name "ClaimDecomp" \
--use_claim ${use_claim} \
--use_decomposed_question ${use_decomposed_question} \
--use_answer ${use_answer} \
--use_bm25_retrieval ${use_bm25_retrieval} \
--use_qa_pairs ${use_qa_pairs} \
--topk_docs ${topk_docs} \
--topk_sents ${topk_sents} \
--num_class ${num_class} \
--use_gpt_rationale ${use_gpt_rationale}

