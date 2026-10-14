# ============================================================
# [讲解] 本文件 = 奖励函数（强化学习里给模型打分的地方），是训练的"指挥棒"。
# [讲解] 它算两类东西再合成一个总分：
# [讲解]   1) 答案对不对：把模型答案和标准答案做文本归一化后，算 F1 或 EM（精确匹配）
# [讲解]   2) 格式&搜索行为对不对：检查输出是否按 <reason>/<search>/<answer> 的格式来，
# [讲解]      并统计搜索了几次；没搜索会被扣分（逼模型学会搜索）
# [讲解] ★改成 ECCS 调度任务时，主要就改这里：
# [讲解]   - 把 cal_f1(答案) 换成你的"调度评分"（可行性 / Gap / 满足率，用 eccs_eval.py 算）
# [讲解]   - 把 is_valid_* 的"答案格式检查"换成"排班输出格式检查"
# ============================================================
import re
import string
import copy
from collections import Counter

import numpy as np


# ============================================================
# Text Normalization & Metrics (from eval.py)
# ============================================================

# [讲解] 文本归一化：转小写、去标点、去冠词(a/an/the)、压空格。
# [讲解] 目的是比较答案时忽略这些无关差异（"The Cat" 和 "cat" 视为相同）。
def normalize_answer(answer: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def calculate_metric_scores_em(gold_answers, predicted_answers, aggregation_fn):
    assert len(gold_answers) == len(predicted_answers)
    example_eval_results = []
    total_em = 0
    for gold_list, predicted in zip(gold_answers, predicted_answers):
        em_scores = [1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0 for gold in gold_list]
        aggregated_em = aggregation_fn(em_scores)
        example_eval_results.append({"ExactMatch": aggregated_em})
        total_em += aggregated_em
    avg_em = total_em / len(gold_answers) if gold_answers else 0.0
    return {"ExactMatch": avg_em}, example_eval_results


def calculate_metric_scores_f1(gold_answers, predicted_answers, aggregation_fn):
    assert len(gold_answers) == len(predicted_answers)

    def compute_f1(gold: str, predicted: str) -> float:
        gold_tokens = normalize_answer(gold).split()
        predicted_tokens = normalize_answer(predicted).split()
        common = Counter(predicted_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(predicted_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        return 2 * (precision * recall) / (precision + recall)

    example_eval_results = []
    total_f1 = 0.0
    for gold_list, predicted in zip(gold_answers, predicted_answers):
        f1_scores = [compute_f1(gold, predicted) for gold in gold_list]
        aggregated_f1 = aggregation_fn(f1_scores)
        example_eval_results.append({"F1": aggregated_f1})
        total_f1 += aggregated_f1
    avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
    return {"F1": avg_f1}, example_eval_results


def cal_em(gold_answers, predicted_answers):
    result, _ = calculate_metric_scores_em(gold_answers, predicted_answers, aggregation_fn=np.max)
    return result["ExactMatch"]


def cal_f1(gold_answers, predicted_answers):
    result, _ = calculate_metric_scores_f1(gold_answers, predicted_answers, aggregation_fn=np.max)
    return result["F1"]


# ============================================================
# Format Validation & Reward Computation (from mmgraph_r1_score.py)
# ============================================================

# [讲解] 从模型输出里用正则抠出 <answer>...</answer> 里的最终答案（取最后一个）。
# [讲解] 抠不到就返回 None（说明模型没按格式给出答案）。
def extract_solution(prediction):
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, prediction, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def is_valid_direct_answer(response, direct_answer_format) -> bool:
    if not re.match(direct_answer_format, response, re.DOTALL):
        return False
    if response.count('<reason>') != 1 or response.count('</reason>') != 1:
        return False
    if response.count('<answer>') != 1 or response.count('</answer>') != 1:
        return False
    if '<search><img></search>' in response:
        return False
    if '<text_search>' in response or '</text_search>' in response:
        return False
    return True


def is_valid_image_search(response, call_image_search_format) -> bool:
    if not re.match(call_image_search_format, response, re.DOTALL):
        return False
    if response.count('<reason>') != 1 or response.count('</reason>') != 1:
        return False
    if response.count('<search><img></search>') != 1:
        return False
    if '<answer>' in response or '</answer>' in response:
        return False
    if '<text_search>' in response or '</text_search>' in response:
        return False
    return True


def is_valid_text_search(response, call_text_search_format) -> bool:
    if not re.match(call_text_search_format, response, re.DOTALL):
        return False
    if response.count('<reason>') != 1 or response.count('</reason>') != 1:
        return False
    if response.count('<text_search>') != 1 or response.count('</text_search>') != 1:
        return False
    if '<answer>' in response or '</answer>' in response:
        return False
    if '<search><img></search>' in response:
        return False
    return True


# [讲解] 检查"多轮输出的格式对不对"，并数搜索了几次。
# [讲解] 规则：多轮时，前面每一轮都必须是合法的"搜索动作"(图搜或文搜)，
# [讲解]       最后一轮必须是合法的"直接回答"。全对 format_score=1，否则 0。
# [讲解] 返回 (format_score 格式分0/1, search_count 搜索次数)。
# [讲解] 三个正则就是三种合法输出模板：直接回答 / 图搜 / 文搜。
def format_reward(input_string: list):
    conv_rounds = len(input_string)
    format_score, search_count = 0, 0

    direct_answer_format = r'^<reason>.*</reason>.*<answer>.*</answer>$'
    call_image_search_format = r'^<reason>.*</reason>.*<search><img></search>$'
    call_text_search_format = r'^<reason>.*</reason>.*<text_search>.*</text_search>$'

    responses = [resp.strip() for resp in input_string]
    if conv_rounds > 1:
        for i in range(conv_rounds - 1):
            if is_valid_image_search(responses[i], call_image_search_format) or is_valid_text_search(responses[i], call_text_search_format):
                search_count += 1
        if search_count == conv_rounds - 1 and is_valid_direct_answer(responses[-1], direct_answer_format):
            format_score = 1
    elif conv_rounds == 1:
        if is_valid_direct_answer(responses[0], direct_answer_format):
            format_score = 1
    else:
        raise ValueError(f"Number of responses is {conv_rounds}, which is not supported!")

    return format_score, search_count


# [讲解] ★核心：把"答案质量"和"格式/搜索行为"合成一个总奖励 overall_score。
# [讲解] 步骤：① 抠出最后一轮的答案 ② 算 F1(答案, 标准答案) 当质量分 score
# [讲解]       ③ 用 format_reward 得到格式分和搜索次数 ④ 按下面三种情况合成总分：
# [讲解]   · 格式对 + 有搜索：  overall = F1 × (1 − 2/(搜索数+2))   ← 正常奖励
# [讲解]   · 格式错 + 有搜索：  overall = −0.5 + F1 × (…)          ← 格式错扣 0.5
# [讲解]   · 完全没搜索：       overall = −0.5                      ← 逼模型必须搜
# [讲解] 系数 (1−2/(n+2)) 随搜索次数增大但边际递减：鼓励多步检索，又不至于"为搜而搜"。
# [讲解] ★改 ECCS：把这里的 cal_f1(...) 换成"排班方案的评分"(可行性/Gap/满足率)即可。
def compute_score(prediction: list, ground_truth: list, raw_response_str: str):
    assert len(prediction) > 0, "Model responses are empty!"
    answer = extract_solution(prediction=prediction[-1])  # [讲解] 取最后一轮的 <answer>

    score = 0
    if answer is not None:
        score = cal_f1([ground_truth], [answer])  # [讲解] 答案质量 = F1（改调度时换成你的评分）

    format_score, search_count = format_reward(prediction)  # [讲解] 格式分 + 搜索次数

    if format_score == 1 and search_count >= 1:
        overall_score = copy.deepcopy(score) * (1 - 2 / (search_count + 2))
    elif format_score == 0 and search_count >= 1:
        overall_score = -0.5 + copy.deepcopy(score) * (1 - 2 / (search_count + 2))
    else:
        overall_score = -0.5

    search_cnt = search_count if format_score == 1 else 0

    return overall_score, score, search_cnt


# ============================================================
# Default compute_score wrapper (from reward_score_mm/__init__.py)
# ============================================================

def _default_compute_score(data_source, solution_str, ground_truth, raw_response_str):
    # [讲解] 按数据源分发：ECCS 调度任务走 score_eccs（可行性/满足率），其余走原 F1 逻辑。
    # [讲解]   ground_truth 对 ECCS 而言是 [实例JSON字符串]（见 build_eccs_parquet 的 candidate_answers），
    # [讲解]   solution_str 是"多轮响应文本列表"，正好当 compute_score_eccs 的 prediction。
    if data_source == "ECCS-RTS":
        import json as _json
        from mmgraph_r1.reward.score_eccs import compute_score_eccs
        instance = _json.loads(ground_truth[0]) if ground_truth else {}
        overall, quality, search_count, _info = compute_score_eccs(solution_str, instance)
        return float(overall), float(quality), float(search_count)

    res, score, search_count = compute_score(solution_str, ground_truth, raw_response_str)
    if isinstance(res, (int, float, bool)):
        return float(res), float(score), float(search_count)
    else:
        return float(res[0]), float(score), float(search_count)
