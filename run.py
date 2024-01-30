import os

ROOT_DIR = os.path.join(os.path.dirname(__file__))

# Disable the TOKENIZERS_PARALLELISM
TOKENIZER_FALSE = "export TOKENIZERS_PARALLELISM=false\n"

# finqa annotation command
os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/annotate_binder_program.py --dataset finqa \
--dataset_split test \
--prompt_file templates/prompts/finqa_binder.txt \
--n_parallel_prompts 1 \
--max_generation_tokens 512 \
--temperature 0.4 \
--generate_type answer \
--sampling_n 20 \
-v""")

# finqa execution command
os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/execute_binder_program.py --dataset finqa \
--dataset_split test \
--qa_retrieve_pool_file templates/qa_retrieve_pool/qa_retrieve_pool.json \
--input_program_file binder_program_finqa_test_chatgpt.json \
--output_program_execution_file binder_program_finqa_test_exec.json \
--vote_method simple""")

# fetaqa annotation command
os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/annotate_binder_program.py --dataset fetaqa \
--dataset_split test \
--prompt_file templates/prompts/fetaqa_binder.txt \
--n_parallel_prompts 1 \
--max_generation_tokens 512 \
--temperature 0.4 \
--generate_type nsql \
--sampling_n 20 \
-v""")

# fetaqa execution command
os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/execute_binder_program.py --dataset fetaqa \
--dataset_split test \
--qa_retrieve_pool_file templates/qa_retrieve_pool/qa_retrieve_pool.json \
--input_program_file binder_program_fetaqa_test_chatgpt.json \
--output_program_execution_file binder_program_fetaqa_test_exec.json \
--vote_method simple""")

# wikitq nsql annotation command
# <<<<<<< HEAD
# os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/annotate_binder_program.py --dataset wikitq \
# --dataset_split test \
# --prompt_file templates/prompts/wikitq_binder.txt \
# --n_parallel_prompts 1 \
# --max_generation_tokens 512 \
# --temperature 0.4 \
# --sampling_n 20 \
# -v""")
# =======
# os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/annotate_binder_program.py --dataset wikitq \
# --dataset_split test \
# --prompt_file templates/prompts/wikitq_binder.txt \
# --n_parallel_prompts 1 \
# --max_generation_tokens 512 \
# --temperature 0.4 \
# --sampling_n 20 \
# # -v""")
# >>>>>>> 45f79534743af891a9788db9375367dbe3851f17

# wikitq nsql execution command
# os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/execute_binder_program.py --dataset wikitq \
# --dataset_split test \
# --qa_retrieve_pool_file templates/qa_retrieve_pool/qa_retrieve_pool.json \
# --input_program_file binder_program_wikitq_test.json \
# --output_program_execution_file binder_program_wikitq_test_exec.json \
# --vote_method simple
# """)

# tab_fact nsql annotation command
# os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/annotate_binder_program.py --dataset tab_fact \
# --dataset_split test \
# --prompt_file templates/prompts/tab_fact_binder.txt \
# --n_parallel_prompts 1 \
# --n_processes 1 \
# --n_shots 18 \
# --max_generation_tokens 256 \
# --temperature 0.6 \
# --sampling_n 50 \
# -v""")

# tab_fact nsql execution command
# os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/execute_binder_program.py --dataset tab_fact \
# --dataset_split test \
# --qa_retrieve_pool_file templates/qa_retrieve_pool/qa_retrieve_pool.json \
# --input_program_file binder_program_tab_fact_test.json \
# --output_program_execution_file binder_program_tab_fact_test_exec.json \
# --allow_none_and_empty_answer \
# --vote_method answer_biased \
# --answer_biased 1 \
# --answer_biased_weight 3 \
# """)

# More
# tab_fact nsql annotation command with example retrieval
# os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/w_ic_examples_retrieval/annotate_binder_program.py --dataset tab_fact \
# --dataset_split test \
# --n_parallel_prompts 1 \
# --n_processes 1 \
# --n_shots 18 \
# --max_generation_tokens 256 \
# --temperature 0.6 \
# --sampling_n 50 \
# -v""")

# mmqa nsql annotation command
# os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/mmqa/annotate_binder_program.py --dataset mmqa \
# --dataset_split validation \
# --prompt_file templates/prompts/mmqa_binder.txt \
# --n_parallel_prompts 1 \
# --n_processes 1 \
# --n_shots 18 \
# --temperature 0.4 \
# --sampling_n 20 \
# -v""")

# mmqa nsql execution command
# os.system(fr"""{TOKENIZER_FALSE}python {ROOT_DIR}/scripts/execute_binder_program.py --dataset mmqa \
# --dataset_split validation \
# --qa_retrieve_pool_file templates/qa_retrieve_pool/mmqa_qa_retrieve_pool.json \
# --input_program_file binder_program_mmqa_validation.json \
# --output_program_execution_file binder_program_mmqa_validation_exec.json \
# --vote_method simple
# """)

# Analysis(Still working on cleaning these code)
# scalability command
# robustness command
# python command