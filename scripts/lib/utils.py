import os
import numpy as np
import sacrebleu
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring
from sklearn.metrics import accuracy_score, f1_score
from multiprocessing import Pool

def split_multi_answer(ans, sep=';', close=True, add_no_comment=False):

    """Splits string of all reference answers into a list of formatted answers"""

    answers = ans.strip().split(sep)
    split_answers = []
    for a in answers:
        a = a.strip()
        if len(a):
            if close:  # add a period after all answers
                if a[-1] != '.':
                    split_answers.append(a + '.')
                else:
                    split_answers.append(a)
            else:
                split_answers.append(a)
    # for generative metrics only, check if "I have no comment" is listed as a correct answer.
    # if not, add it.
    if add_no_comment and "I have no comment." not in split_answers:
        split_answers.append("I have no comment.")

    return split_answers

def bleurt_score(prediction, ref_true, ref_false, bleurt):
    assert len(prediction) == len(ref_true) == len(ref_false)
    res_metric = {}
    for idx in tqdm(range(len(prediction))):
        scores_true = bleurt.compute(predictions=[prediction[idx]] * len(ref_true[idx]),
                                     references=ref_true[idx])['scores']
        scores_false = bleurt.compute(predictions=[prediction[idx]] * len(ref_false[idx]),
                                      references=ref_false[idx])['scores']
        for calc in ['max', 'diff', 'acc']:
            col_name = f'BLEURT_{calc}'
            if col_name not in res_metric:
                res_metric[col_name] = []

            if calc == 'max':
                res_metric[col_name].append(max(scores_true))
            elif calc == 'diff':
                res_metric[col_name].append(max(scores_true) - max(scores_false))
            elif calc == 'acc':
                res_metric[col_name].append(int(max(scores_true) > max(scores_false)))
    
    for key in res_metric:
        res_metric[key] = np.mean(res_metric[key])
    
    return res_metric

def tmp_func(a, b, bleurt):
    return bleurt.compute(predictions=a,references=b)['scores']

def bleurt_score_parallel(prediction, ref_true, ref_false, bleurt):
    assert len(prediction) == len(ref_true) == len(ref_false)
    res_metric = {}
    
    pool = Pool(os.cpu_count())
    
    mp_list = [([prediction[idx]] * len(ref_true[idx]), ref_true[idx], bleurt) for idx in range(len(prediction))]
    mapping = pool.starmap(tmp_func, mp_list)
    scores_true_list = [tmp for tmp in mapping]

    mp_list = [([prediction[idx]] * len(ref_false[idx]), ref_false[idx], bleurt) for idx in range(len(prediction))]
    mapping = pool.starmap(tmp_func, mp_list)
    scores_false_list = [tmp for tmp in mapping]

    for idx in range(len(prediction)):
        #scores_true = bleurt.compute(predictions=[prediction[idx]] * len(ref_true[idx]),
        #                             references=ref_true[idx])['scores']
        #scores_false = bleurt.compute(predictions=[prediction[idx]] * len(ref_false[idx]),
        #                              references=ref_false[idx])['scores']
        scores_true = scores_true_list[idx]
        scores_false = scores_false_list[idx]

        for calc in ['max', 'diff', 'acc']:
            col_name = f'BLEURT_{calc}'
            if col_name not in res_metric:
                res_metric[col_name] = []

            if calc == 'max':
                res_metric[col_name].append(max(scores_true))
            elif calc == 'diff':
                res_metric[col_name].append(max(scores_true) - max(scores_false))
            elif calc == 'acc':
                res_metric[col_name].append(int(max(scores_true) > max(scores_false)))
    
    for key in res_metric:
        res_metric[key] = np.mean(res_metric[key])
    
    return res_metric

def bleu_score(prediction, ref_true, ref_false, return_error_idx=False):
    assert len(prediction) == len(ref_true) == len(ref_false)
    res_metric = {}
    pool = Pool(os.cpu_count())
    for idx in range(len(prediction)):
        all_refs = ref_true[idx] + ref_false[idx]
        #bleu_scores = [_bleu([ref], [prediction[idx]]) for ref in all_refs]
        mp_list = [([ref], [prediction[idx]]) for ref in all_refs]
        mapping = pool.starmap(_bleu, mp_list)
        bleu_scores = [tmp for tmp in mapping]
        bleu_correct = np.nanmax(bleu_scores[:len(ref_true[idx])])
        bleu_incorrect = np.nanmax(bleu_scores[len(ref_true[idx]):])

        for calc in ['max', 'diff', 'acc']:
            col_name = f'BLEU_{calc}'
            if col_name not in res_metric:
                res_metric[col_name] = []
            
            if calc == 'max':
                res_metric[col_name].append(bleu_correct)
            elif calc == 'diff':
                res_metric[col_name].append(bleu_correct - bleu_incorrect)
            elif calc == 'acc':
                res_metric[col_name].append(int(bleu_correct > bleu_incorrect))
    
    error_idx = [i for i, value in enumerate(res_metric['BLEU_acc']) if value == 0]
    
    for key in res_metric:
        res_metric[key] = np.mean(res_metric[key])

    if return_error_idx:
        res_metric["BLEU_error_idx"] = error_idx
    
    return res_metric


def rouge_score(prediction, ref_true, ref_false):
    assert len(prediction) == len(ref_true) == len(ref_false)
    res_metric = {}
    pool = Pool(os.cpu_count())
    for idx in range(len(prediction)):
        all_refs = ref_true[idx] + ref_false[idx]
        #rouge_scores = [_rouge([ref], [prediction[idx]]) for ref in all_refs]
        mp_list = [([ref], [prediction[idx]]) for ref in all_refs]
        mapping = pool.starmap(_rouge, mp_list)
        rouge_scores = [tmp for tmp in mapping]
        
        for score_type in ['rouge1', 'rouge2', 'rougeLsum']:
            for calc in ['max', 'diff', 'acc']:
                rouge_scores_of_type = [score[score_type] for score in rouge_scores]
                rouge_correct = np.nanmax(rouge_scores_of_type[:len(ref_true[idx])])
                rouge_incorrect = np.nanmax(rouge_scores_of_type[len(ref_true[idx]):])
                col_name = f'{score_type}_{calc}'
                if col_name not in res_metric:
                    res_metric[col_name] = []
                
                if calc == 'max':
                    res_metric[col_name].append(rouge_correct)
                elif calc == 'diff':
                    res_metric[col_name].append(rouge_correct - rouge_incorrect)
                elif calc == 'acc':
                    res_metric[col_name].append(int(rouge_correct > rouge_incorrect))

    for key in res_metric:
        res_metric[key] = np.mean(res_metric[key])
    
    return res_metric


def _bleu(refs, preds):
    """
    Returns `t5` style BLEU scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L41

    :param refs:
        A `list` of `list` of reference `str`s.
    :param preds:
        A `list` of predicted `str`s.
    """
    if isinstance(refs[0], list):
        refs = [[x for x in ref] for ref in refs]
    else:
        # Need to wrap targets in another list for corpus_bleu.
        refs = [refs]
    score = sacrebleu.corpus_bleu(
        preds,
        refs,
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False,
    ).score
    return score


def _rouge(refs, preds):
    """
    Returns `t5` style ROUGE scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68

    :param refs:
        A `list` of reference `strs`.
    :param preds:
        A `list` of predicted `strs`.
    """
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(rouge_types)
    # Add newlines between sentences to correctly compute `rougeLsum`.

    def _prepare_summary(summary):
        summary = summary.replace(" . ", ".\n")
        return summary

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        ref = _prepare_summary(ref)
        pred = _prepare_summary(pred)
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}

def custom_f1_score(true_label_list, pred_label_list, model_name=""):
    error_num = 0
    full_true_label_list, full_pred_label_list = [], []
    no_error_true_label_list, no_error_pred_label_list = [], []
    for idx in range(len(pred_label_list)):
        if pred_label_list[idx] not in [0, 1]:
            full_true_label_list.append(true_label_list[idx])
            full_pred_label_list.append(0)
            error_num += 1
        else:
            full_true_label_list.append(true_label_list[idx])
            full_pred_label_list.append(pred_label_list[idx])
            no_error_true_label_list.append(true_label_list[idx])
            no_error_pred_label_list.append(pred_label_list[idx])

    metrics = {f"{model_name}_f1": f1_score(full_true_label_list, full_pred_label_list, zero_division=0.0),}
            #f"{model_name}_f1_no_error": f1_score(no_error_true_label_list, no_error_pred_label_list, zero_division=0.0),
            #f"{model_name}_error": error_num}
    return metrics


from latex2sympy2_extended import NormalizationConfig, normalize_latex
from math_verify import LatexExtractionConfig, StringExtractionConfig, parse, verify
import re

def _additional_normalization(expr):
    # Remove % and \\% from the number
    percentage_pattern = r"^(\d+\.?\d*)(?:\\%|%)$"
    match_gt = re.fullmatch(percentage_pattern, expr)
    if match_gt:
        expr = match_gt.group(1)
    # Remove . corresponding to the end of sentence
    expr = expr.rstrip(".\\")
    return expr

def math_equal(gt_answer, predicted_answer, take_modulo: int | None = None, **kwargs):
    if predicted_answer is None:
        return False

    gt_answer = str(gt_answer)
    predicted_answer = str(predicted_answer)

    # if we are sure that gt is always integer
    if take_modulo is not None:
        gt_answer = int(gt_answer) % take_modulo
        try:
            predicted_answer = int(predicted_answer) % take_modulo
        except:
            predicted_answer = None
        # no need to simpy call in this case
        return predicted_answer == gt_answer

    # Try to compare as MCQ options
    mcq_options = "ABCDEFGHIJ"
    norm_gt_mcq = gt_answer.strip()

    is_mcq = re.fullmatch("|".join(mcq_options), norm_gt_mcq)
    parsed_gt = parse(gt_answer, [StringExtractionConfig(strings=tuple(mcq_options))])
    parsed_pred = parse(predicted_answer, [StringExtractionConfig(strings=tuple(mcq_options))])
    if is_mcq and verify(parsed_gt, parsed_pred):
        return verify(parsed_gt, parsed_pred)

    # Additional normalization step
    gt_answer = _additional_normalization(gt_answer)
    predicted_answer = _additional_normalization(predicted_answer)

    # Try literal comparison
    literal_pattern = r"[a-zA-Z ,]+|[0-9 ]+"
    normalized_gt = normalize_latex(gt_answer, NormalizationConfig)
    normalized_pred = normalize_latex(predicted_answer, NormalizationConfig)
    is_literal = re.fullmatch(literal_pattern, normalized_gt) and re.fullmatch(literal_pattern, normalized_pred)
    is_normalized_equal = normalized_gt.replace(" ", "") == normalized_pred.replace(" ", "")

    if is_literal or is_normalized_equal:
        return is_normalized_equal

    # Fallback to symbolic comparison
    current_gt_answer = gt_answer
    current_predicted_answer = predicted_answer

    # math_verify.parse expects input to be in latex environment, e.g. $...$
    latex_env_search_pattern = r"\$.*\$|\\\(.*\\\)|\\\[.*\\\]|\\boxed\{"
    if not re.search(latex_env_search_pattern, current_gt_answer, re.DOTALL):
        current_gt_answer = f"${current_gt_answer}$"
    if not re.search(latex_env_search_pattern, current_predicted_answer, re.DOTALL):
        current_predicted_answer = f"${current_predicted_answer}$"

    parsed_gt = parse(current_gt_answer, [LatexExtractionConfig()])
    parsed_pred = parse(current_predicted_answer, [LatexExtractionConfig()])

    return verify(parsed_gt, parsed_pred, **kwargs)


# 提取 boxed{} 内容
def extract_boxed_answer(text):
    start_token = r"\boxed{"
    start_idx = text.rfind(start_token)
    if start_idx == -1:
        return ''
    i = start_idx + len(start_token)
    brace_depth = 1
    content = []
    while i < len(text):
        if text[i] == '{':
            brace_depth += 1
        elif text[i] == '}':
            brace_depth -= 1
            if brace_depth == 0:
                break
        content.append(text[i])
        i += 1
    return ''.join(content).strip() if brace_depth == 0 else ''