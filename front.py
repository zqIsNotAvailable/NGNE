import argparse
import copy
import re
import time
import os
from nsql.database import NeuralDB
from utils.utils import load_data_split, majority_vote
import streamlit as st
import pandas as pd
from generation.generator import Generator
from transformers import AutoTokenizer
from nsql.nsql_exec import Executor, NeuralDB
from nsql.nsql_exec_python import Executor as Executor_Python
from nsql.nsql_exec_fin_answer import Executor as Executor_Answer
from utils.normalizer import post_process_sql
from utils.evaluator import Evaluator
st.set_page_config(page_title="binder", page_icon=":clipboard:")
ROOT_DIR = os.path.join(os.path.dirname(__file__), "./")


def extract_questions(d):
    if isinstance(d, dict):
        return [value for key, value in d.items() if 'question' in key][0]
    return []


def extract_answers(d):
    if isinstance(d, dict):
        return [value for key, value in d.items() if 'exe_ans' in key][0]
    return []


@st.cache_data
def load_dataset_in_df(bd, ds):
    dataset = load_data_split(bd, ds)
    return pd.DataFrame(dataset), dataset.info.description


with open("key.txt", 'r') as f:
    keys = [line.strip() for line in f.readlines()]

parser = argparse.ArgumentParser()
# Binder program generation options
parser.add_argument('--prompt_style', type=str, default='create_table_select_3_full_table',
                    choices=['create_table_select_3_full_table',
                             'create_table_select_full_table',
                             'create_table_select_3',
                             'create_table',
                             'create_table_select_3_full_table_w_all_passage_image',
                             'create_table_select_3_full_table_w_gold_passage_image',
                             'no_table'])
# Codex options
parser.add_argument('--engine', type=str, default="gpt-3.5-turbo")
parser.add_argument('--max_generation_tokens', type=int, default=256)
parser.add_argument('--temperature', type=float, default=0.4)
parser.add_argument('--sampling_n', type=int, default=5)
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--stop_tokens', type=str, default='\n\n',
                    help='Split stop tokens by ||')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--qa_retrieve_pool_file', type=str, default='templates/qa_retrieve_pool/qa_retrieve_pool.json')
parser.add_argument('--allow_none_and_empty_answer', action='store_true',
                    help='Whether regarding none and empty executions as a valid answer.')
parser.add_argument('--answer_placeholder', type=int, default=0,
                    help='Placeholder answer if execution error occurs.')
parser.add_argument('--vote_method', type=str, default='simple',
                    choices=['simple', 'prob', 'answer_biased'])
parser.add_argument('--answer_biased', type=int, default=None,
                    help='The answer to be biased w. answer_biased_weight in majority vote.')
parser.add_argument('--answer_biased_weight', type=float, default=None,
                    help='The weight of the answer to be biased in majority vote.')
parser.add_argument('--process_program_with_fuzzy_match_on_db', action='store_false',
                    help='Whether use fuzzy match with db and program to improve on program.')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

generator = Generator(args, keys=keys)


st.header('binder')
st.subheader('dataset')
container_dataset = st.empty()
container_split = st.empty()
binder_dataset = container_dataset.selectbox("Please select the dataset", ["wikitq", "fetaqa", "finqa"])
prompt_file = "templates/prompts/" + binder_dataset + "_binder.txt"


dataset_select = container_split.selectbox("please select the split type", ["train", "test", "validation"])
st.write("the dataset you selected:", binder_dataset)
start_time = time.time()
df, di = load_dataset_in_df(binder_dataset, dataset_select)
st.caption("description")
st.write(di)
st.subheader("Sample Data of dataset")
# st.write(df.head())
st.write(df)
st.subheader("running setting")
running_mode = st.selectbox("please select the running mode", ["sample", "whole"])


is_question_in = True
answer_texts = []
questions = []
table_filtered = []
filtered_df = None
pre_text = ""
post_text = ""

if running_mode == "sample":
    if binder_dataset != "finqa":
        codex_language = st.selectbox("please select a target binder program", ["Binder-SQL", "Binder-Python"])
        generate_type = 'nsql'
    else:
        codex_language = st.selectbox("please select a target binder program", ["Binder-QA"])
        generate_type = 'answer'

    if codex_language == "Binder-Python":
        generate_type = 'npython'
        prompt_file = 'templates/prompts/prompt_wikitq_python_simplified_v4.txt'
    # table_method = st.selectbox("please select the method of choosing the table", ["manual sample", "automatic sample"])
    # if table_method == "manual sample":
    if binder_dataset == "fetaqa":
        table_choose = st.selectbox("please select a table", df['table_page_title'].tolist())
        table = df[df['table_page_title'] == table_choose]
        table_section = st.selectbox("please select a section of the table", set(table['table_section_title'].tolist()))
        table = df[df['table_page_title'] == table_choose][df['table_section_title'] == table_section]
        questions = table["question"].tolist()
        answer_texts = table["answer"].tolist()
        questions.append("others")
        page_title = table_choose
        feta_page_title = page_title
        section_title = table_section
        unit_table = table.iloc[0]['table_array']

        columns = unit_table[0]
        duplicates = [item for item in set(columns) if columns.count(item) > 1]

        for column_name in duplicates:
            indices = [i for i, item in enumerate(columns) if item == column_name]
            for index, i in enumerate(indices[1:], 1):
                columns[i] = f"{column_name}_{index}"

        unit_table_rows = [unit_table[i] for i in range(1, len(unit_table))]
        feta_header = columns
        feta_rows = unit_table_rows
        df_table = pd.DataFrame(unit_table_rows, columns=columns)
    elif binder_dataset == "finqa":
        id_4_table_name = df['id'].tolist()
        unique_id_4_table_name = set()
        for item in id_4_table_name:

            element = item.split('-')[0]
            unique_id_4_table_name.add(element)

        unique_list = list(unique_id_4_table_name)
        table_choose = st.selectbox("please select a table", unique_list)
        table = df[df['id'].str.contains(table_choose)]
        page_title = table_choose
        questions = table['qa'].apply(extract_questions).tolist()
        answer_texts = table['qa'].apply(extract_answers).tolist()
        questions.append("others")
        unit_table = table.iloc[0]['table']

        pre_text = table.iloc[0]['pre_text']
        post_text = table.iloc[0]['post_text']

        columns = unit_table[0]
        duplicates = [item for item in set(columns) if columns.count(item) > 1]

        for column_name in duplicates:
            indices = [i for i, item in enumerate(columns) if item == column_name]
            for index, i in enumerate(indices[1:], 1):
                columns[i] = f"{column_name}_{index}"

        unit_table_rows = [unit_table[i] for i in range(1, len(unit_table))]
        fin_header = unit_table[0]
        fin_rows = unit_table_rows

        df_table = pd.DataFrame(fin_rows, columns=fin_header)
    else:
        table_choose = st.selectbox("please select a table", df['table'].apply(lambda x: x["page_title"]).tolist())
        table = df['table'].apply(lambda x: x if x.get("page_title") == table_choose else None)
        filtered_df = df[df['table'].apply(lambda x: x.get('page_title') == table_choose)]
        table_filtered = [d for d in table if d is not None]

        questions = filtered_df["question"].tolist()
        answer_texts = filtered_df["answer_text"].tolist()
        questions.append("others")

        header = table_filtered[0]["header"]
        page_title = table_filtered[0]["page_title"]
        rows = table_filtered[0]["rows"]
        duplicates = [item for item in set(header) if header.count(item) > 1]

        for column_name in duplicates:
            indices = [i for i, item in enumerate(header) if item == column_name]
            for index, i in enumerate(indices[1:], 1):
                header[i] = f"{column_name}_{index}"

        df_table = pd.DataFrame(rows, columns=header)

    st.caption("display the table")
    st.write("Title:", page_title)
    if binder_dataset == "fetaqa":
        st.write("Section:", section_title)
    st.write(df_table)

    question = st.selectbox("please ask a question", questions)
    if question == "others":
        question = st.text_input("ask anything about the table")
        is_question_in = False

    n_shots = st.number_input('Insert a number for n shot', 0, 13, 8)

    if st.button('Run Binder!'):

        st.subheader("Question")
        st.write(question)
        if is_question_in:
            st.write("ground truth:", answer_texts[questions.index(question)])

        built_few_shot_prompts = []
        if binder_dataset == 'fetaqa':
            g_data_item = {'table': {'page_title': page_title, 'header': feta_header, 'rows': feta_rows}}

        elif binder_dataset == 'finqa':
            g_data_item = {'table': {'page_title': page_title, 'header': fin_header, 'rows': fin_rows}, 'pre_text': pre_text, 'post_text': post_text}

        else:
            g_data_item = filtered_df.iloc[0].to_dict()

        g_data_item["question"] = question
        g_dict = {
            'generations': [],
            'ori_data_item': copy.deepcopy(g_data_item)
        }

        db = NeuralDB(
            tables=[{'title': g_data_item['table']['page_title'], 'table': g_data_item['table']}]
        )
        g_data_item['table'] = db.get_table_df()
        g_data_item['title'] = db.get_table_title()

        few_shot_prompt = generator.build_few_shot_prompt_from_file(
            file_path=prompt_file,
            n_shots=n_shots
        )
        generate_prompt = generator.build_generate_prompt(
            data_item=g_data_item,
            generate_type=(generate_type,)
        )
        prompt = few_shot_prompt + "\n\n" + generate_prompt
        # print(os.path.join(ROOT_DIR, "utils", "gpt2"))

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(ROOT_DIR, "utils", "gpt2"))

        # Ensure the input length fit Codex max input tokens by shrinking the n_shots
        # max_api minus max_generation
        max_prompt_tokens = 3800 - 512
        while len(tokenizer.tokenize(prompt)) >= max_prompt_tokens:
            n_shots -= 1
            assert n_shots >= 0
            few_shot_prompt = generator.build_few_shot_prompt_from_file(
                file_path=prompt_file,
                n_shots=n_shots
            )
            prompt = few_shot_prompt + "\n\n" + generate_prompt

        built_few_shot_prompts.append((1, prompt))

        response_dict = generator.generate_one_pass(
            prompts=built_few_shot_prompts
        )

        st.write(response_dict)
        result_dict = dict()
        result_dict['question'] = question
        if is_question_in:
            result_dict['gold_answer'] = answer_texts[questions.index(question)]

        executor = Executor(args, keys)
        executor_python = Executor_Python(args, keys)
        executor_answer = Executor_Answer()
        # Execute
        exec_answer_list = []
        nsql_exec_answer_dict = dict()

        for idx, (nsql, logprob) in enumerate(response_dict[1]):

            try:
                if nsql in nsql_exec_answer_dict:
                    exec_answer = nsql_exec_answer_dict[nsql]
                else:

                    if generate_type == 'nsql':
                        nsql = post_process_sql(
                            sql_str=nsql,
                            df=db.get_table_df(),
                            process_program_with_fuzzy_match_on_db=args.process_program_with_fuzzy_match_on_db,
                            table_title=g_data_item['title']
                        )

                        exec_answer = executor.nsql_exec(nsql, db, verbose=args.verbose)

                    elif generate_type == 'npython':

                        lines = nsql.split('\n')
                        lines.insert(1, "    table.columns = table.columns.str.lower()")
                        nsql = '\n'.join(lines)

                        lines = nsql.split('\n')
                        for i in range(1, len(lines)):
                            lines[i] = lines[i].replace('==', '.str.lower() ==')

                        nsql = '\n'.join(lines)

                        exec_answer = executor_python.nsql_exec(nsql, df_table, verbose=args.verbose)

                    elif generate_type == 'answer':
                        exec_answer = executor_answer.nsql_exec(nsql.replace('%', '').replace('$', ''))

                    if isinstance(exec_answer, str):
                        exec_answer = [exec_answer]
                    nsql_exec_answer_dict[nsql] = exec_answer
                exec_answer_list.append(exec_answer)
            except Exception as e:
                exec_answer = '<error>'
                exec_answer_list.append(exec_answer)

            if response_dict.get('exec_answers', None) is None:
                response_dict['exec_answers'] = []
            response_dict['exec_answers'].append(exec_answer)

        pred_answer, pred_answer_nsqls = majority_vote(
            nsqls=response_dict[1],
            pred_answer_list=exec_answer_list,
            answer_placeholder=args.answer_placeholder,
            vote_method=args.vote_method,
            answer_biased=args.answer_biased,
            answer_biased_weight=args.answer_biased_weight
        )
        # Evaluate
        result_dict['pred_answer'] = pred_answer
        result_dict['nsql'] = pred_answer_nsqls

        if pred_answer == 0:
            st.write("fail to predict")
        else:
            st.write(f'answer: {pred_answer}')
        if is_question_in:
            st.write('evaluation:')

            score = Evaluator().evaluate(
                pred_answer,
                answer_texts[questions.index(question)],
                dataset=binder_dataset,
                question=result_dict['question']
            )

            if binder_dataset == 'fetaqa':

                st.write("sacreBLEU:")
                st.write(score[0])
                st.write("BERTScore:")
                st.write(score[1])
                st.write("ROUGE-1:")
                st.write(score[2])
                st.write("ROUGE-2:")
                st.write(score[3])
                st.write("ROUGE-L:")
                st.write(score[4])
                st.write("GPT-evaluation:")
                st.write(score[5])
            else:
                if score == 0:
                    st.write('Wrong')
                else:
                    st.write('Correct!')
else:
    container_dataset.empty()
    binder_dataset = container_dataset.selectbox("Please select the dataset", ["fetaqa", "finqa"])
    container_split.empty()
    dataset_select = container_split.selectbox("please select the split type", ["test"])
    overall_accuracy = None
    st.subheader('Overall Accuracy')
    if binder_dataset == 'fetaqa':

        file_path = 'results/feta_res'
    else:
        file_path = 'results/fin_res'

    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Overall Accuracy: ' in line:
                    colon_index = line.find(":")
                    if colon_index != -1:
                        overall_accuracy = line[colon_index + 1:].strip()

    except FileNotFoundError:
        print("please generate the file feta_res first")

    if binder_dataset == 'fetaqa':
        matches = [num for num in overall_accuracy[1:-1].split(',')]
        st.write("sacreBLEU:")
        st.write(matches[0])
        st.write("ROUGE-1:")
        st.write(matches[1])
        st.write("ROUGE-2:")
        st.write(matches[2])
        st.write("ROUGE-L:")
        st.write(matches[3])
        st.write("GPT-evaluation:")
        st.write(matches[4])

    else:
        st.write(overall_accuracy)



