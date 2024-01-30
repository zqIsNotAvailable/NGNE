import re
import openai
from utils.normalizer import str_normalize
from utils.wtq.evaluator import to_value_list, check_denotation
from utils.mmqa.evaluator import acc
from sacrebleu.metrics import BLEU
from bert_score import score
from rouge_score import rouge_scorer
import os



class Evaluator:
    def __init__(self, keys=None, from_front=1):
        if from_front == 0:
            self.keys = keys
            self.current_key_id = 0
            self.api_key = self.keys[self.current_key_id]
            self.current_key_id = (self.current_key_id + 1) % len(self.keys)
        elif from_front == 1:
            current_directory = os.path.dirname(__file__)
            parent_directory = os.path.dirname(current_directory)
            file_path = os.path.join(parent_directory, 'key.txt')
            with open(file_path, "r") as file:
                self.api_key = file.readline().strip()


    def evaluate(
            self,
            pred_answer,
            gold_answer,
            dataset,
            allow_semantic=True,
            question=None,
            fetachoice=0
    ):
        if dataset == 'wikitq':
            return self.eval_ex_match(pred_answer, gold_answer, allow_semantic, question)
        elif dataset == 'tab_fact':
            return self.eval_tabfact_match(pred_answer, gold_answer)
        elif dataset == 'mmqa':
            # For more metrics on MMQA,
            # please use the utils/mmqa/eval_mmqa.py to call official on all prediction data
            return self.eval_mmqa_match(pred_answer, gold_answer)
        elif dataset == 'finqa':
            return self.eval_fin_match(pred_answer, gold_answer)
        elif dataset == 'fetaqa':
            return self.eval_feta_match(pred_answer, gold_answer, question, fetachoice)
        else:
            raise ValueError(f'{dataset} evaluator is not supported.')

    def calculate_correctness_score(self, question, ground_truth, prediction):
        openai.api_key = self.api_key
        # Construct the full prompt
        full_prompt = f"Compare the ground truth and prediction from AI models, to give a correctness score for the prediction.\
    The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,\
    0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score.\
    Question | Ground truth | Prediction | Correctness\
    — | — | — | —\
    Who won the 1982 Illinois gubernatorial election, and how many votes was the margin? | Thompson prevailed in the 1982 Illinois gubernatorial election by a 5,074 vote margin. | republican hold, swing | 0.0\
    What are the download rates of EUTRAN? | EUTRAN has download rates of 299.6 Mbit/s and 150.8 Mbit/s. | 0.68, 1.0, 1.0, 10.3, 51.0, 102.0, 150.8, 299.6, 301.5, 301.5, 2998.6, 452.2, 452.2, 603.0, 603.0, 391.7, 391.7, 750.0, 979.0, 25065.0, 1174.0, 1566.0 | 0.1\
    Who won the 400 m final at the 015 Asian Athletics Championship? | Abdalelah Haroun held the 2015 Asian Athletics Championship in the 400 m final with his 44.68. | 44.68000000000000004218847494 | 0.3\
    When and in what play did Platt appear at the Music Box Theatre? | In 2016 and 2017, Platt played in Dear Evan Hansen on Broadway at the Music Box Theatre. | 2000, dear evan hansen | 0.5\
    In which two events does Macarena Reyes hold Chilean records? | Macarena Reyes holds Chilean records in the long jump and heptathlon. | long jump, swimming | 0.5\
    When and in what play did Platt appear at the Music Box Theatre? | In 2016 and 2017, Platt played in Dear Evan Hansen on Broadway at the Music Box Theatre. | 2016–2017, dear evan hansen | 1.0\
    In which two events does Macarena Reyes hold Chilean records? | Macarena Reyes holds Chilean records in the long jump and heptathlon. | long jump, heptathlon | 1.0" + question + " | " + ground_truth + " | " + prediction + " |"

        # Use the OpenAI GPT-3 API to generate a response
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=full_prompt,
            max_tokens=150
        )

        # Extract the generated correctness score from the API response
        generated_output = response.choices[0].text.strip()
        try:
            correctness_score = float(generated_output)
        except ValueError:
            return 0
        return correctness_score

    def eval_ex_match(self, pred, gold, allow_semantic=True, question=None):
        if not isinstance(pred, list):
            pred = [pred]
            gold = [gold]

        pred = [str(p).lower().strip() for p in pred]
        gold = [str(g).lower().strip() for g in gold]

        if not allow_semantic:
            # WikiTQ eval w. string normalization using recognizer
            pred = [str_normalize(span) for span in pred]
            gold = [str_normalize(span) for span in gold]
            pred = to_value_list(pred)
            gold = to_value_list(gold)
            return check_denotation(pred, gold)
        else:
            assert isinstance(question, str)
            question = re.sub('\s+', ' ', question).strip().lower()
            pred = [str_normalize(span) for span in pred]
            gold = [str_normalize(span) for span in gold]
            pred = sorted(list(set(pred)))
            gold = sorted(list(set(gold)))
            # (1) 0 matches 'no', 1 matches 'yes'; 0 matches 'more', 1 matches 'less', etc.
            if len(pred) == 1 and len(gold) == 1:
                if (pred[0] == '0' and gold[0] == 'no') \
                        or (pred[0] == '1' and gold[0] == 'yes'):
                    return True
                question_tokens = question.split()
                try:
                    pos_or = question_tokens.index('or')
                    token_before_or, token_after_or = question_tokens[pos_or - 1], question_tokens[pos_or + 1]
                    if (pred[0] == '0' and gold[0] == token_after_or) \
                            or (pred[0] == '1' and gold[0] == token_before_or):
                        return True
                except Exception as e:
                    pass
            # (2) Number value (allow units) and Date substring match
            if len(pred) == 1 and len(gold) == 1:
                NUMBER_UNITS_PATTERN = re.compile('^\$*[+-]?([0-9]*[.])?[0-9]+(\s*%*|\s+\w+)$')
                DATE_PATTERN = re.compile('[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})?')
                DURATION_PATTERN = re.compile('(P|PT)(\d+)(Y|M|D|H|S)')
                p, g = pred[0], gold[0]
                # Restore `duration` type, e.g., from 'P3Y' -> '3'
                if re.match(DURATION_PATTERN, p):
                    p = re.match(DURATION_PATTERN, p).group(2)
                if re.match(DURATION_PATTERN, g):
                    g = re.match(DURATION_PATTERN, g).group(2)
                match = False
                num_flag, date_flag = False, False
                # Number w. unit match after string normalization.
                # Either pred or gold being number w. units suffices it.
                if re.match(NUMBER_UNITS_PATTERN, p) or re.match(NUMBER_UNITS_PATTERN, g):
                    num_flag = True
                # Date match after string normalization.
                # Either pred or gold being date suffices it.
                if re.match(DATE_PATTERN, p) or re.match(DATE_PATTERN, g):
                    date_flag = True
                if num_flag:
                    p_set, g_set = set(p.split()), set(g.split())
                    if p_set.issubset(g_set) or g_set.issubset(p_set):
                        match = True
                if date_flag:
                    p_set, g_set = set(p.replace('-', ' ').split()), set(g.replace('-', ' ').split())
                    if p_set.issubset(g_set) or g_set.issubset(p_set):
                        match = True
                if match:
                    return True
            pred = to_value_list(pred)
            gold = to_value_list(gold)
            return check_denotation(pred, gold)

    def eval_tabfact_match(self, pred, gold):
        if isinstance(pred, list):
            pred = pred[0]
        pred, gold = str(pred), str(gold)
        return pred == gold

    def eval_mmqa_match(self, pred_answer, gold_answer):
        return acc(pred_answer, gold_answer)

    def eval_fin_match(self, pred_answer, gold_answer):
        if pred_answer == 0:
            return False
        try:
            pred_float = float(pred_answer[0])
            gold_float = float(gold_answer)
        except ValueError:
            if pred_answer[0].lower() == gold_answer.lower():
                return True
            return False

        if abs(pred_float - gold_float) < 0.005 or abs(pred_float * 100 - gold_float) < 0.005 or abs(pred_float / 100 - gold_float) < 0.005:
            return True

        return False

    def eval_feta_match(self, pred_answer, gold_answer, question, fetachoice):
        # sacrebleu

        refs = [gold_answer]
        if pred_answer == 0:
            sys = "0"
        else:
            sys = str(pred_answer[0])

            for i in range(1, len(pred_answer)):
                sys += ", " + str(pred_answer[i])

        bleu = BLEU(lowercase=True, effective_order=True)

        # bertscore
        if fetachoice == 0:
            bertscore, _, _ = score(cands=[sys], refs=[refs], lang='en')

        # rouge
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(gold_answer, sys)

        # gpt
        correctness_score = self.calculate_correctness_score(question, gold_answer, sys)
        if fetachoice == 0:
            return bleu.sentence_score(sys, refs).score, bertscore.item(), scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure, correctness_score
        return bleu.sentence_score(sys, refs).score, scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure, correctness_score



