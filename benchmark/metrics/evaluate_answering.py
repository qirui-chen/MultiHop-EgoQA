import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from answering_metrics import (Bleu, Cider, Meteor, Rouge, SentenceTransformerSimilarity)
from openai import OpenAI
from tqdm import tqdm


PROMPT = """As the instructor of a video understanding course, you have assigned your students to watch a first-person perspective video and answer a question related to its content. Your task is to grade students' answer based on the question and reference answer. Your score should be between 1 and 10, with a higher score indicating a closer match between the student's answer and the reference answer. When grading, consider the following aspects of the student's answer: helpfulness, relevance, accuracy, and level of detail. Provide a brief rationale for your score.

Ensure your response is a dict with 'score' and 'rationale' as keys. For example, {eg}.

###Question
{question}

###Reference Answer
{ref}

###Student Answer
{pred}
"""


def get_response_gpt4(client, question, ref_answer, pred_answer):

    full_prompt = PROMPT.format(eg="{'score': 5, 'rationale': '<rationale>'}",
                                question=question,
                                ref=ref_answer,
                                pred=pred_answer)
    # print(full_prompt)
    # exit(0)
    message = [{
        "role": "system",
        "content": "You are a helpful assistant designed to output JSON."
    }, {
        "role": "user",
        "content": full_prompt
    }]
    response = client.chat.completions.create(
        model='gpt-4o-2024-05-13',
        response_format={"type": "json_object"},
        messages=message,
        max_tokens=256,
        temperature=0.2,
    )

    response = eval(response.choices[0].message.content)
    if 'messages' in response:
        response = eval(response['messages'][0]['content'])
    # print(response)
    return response


def parse_response(response):
    # print(response)
    score = response['score']
    rationale = response['rationale']
    return score, rationale


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--pred_file', type=str, default="")
    parser.add_argument('--gt_file', type=str, default="")
    parser.add_argument('--results_file', type=str, default="")
    args = parser.parse_args()

    client = OpenAI(base_url="", api_key=os.getenv('OPENAI_API_KEY'))

    predictions = json.load(open(args.pred_file))
    labels = json.load(open(args.gt_file))
    # assert len(predictions) == len(labels)

    bleu4_scorer = Bleu(4)
    meteor_scorer = Meteor()
    rouge_scorer = Rouge()
    cider_scorer = Cider()
    similarity_scorer = SentenceTransformerSimilarity()
    all_pred_answers, all_gt_answers, id2idx = {}, {}, {}
    for idx, sample in enumerate(predictions):
        label = labels[idx]
        for item in labels:
            if item['sample_id'] == sample['sample_id']:
                label = item
        assert sample['sample_id'] == label['sample_id'], "{} and {}".format(sample['sample_id'], label['sample_id'])
        pred_answer = sample['A']
        ref_answer = label['A']
        # skip when questions
        if re.search(r"<T\d+></T\d+>", ref_answer):
            continue
        # remove special tokens
        ref_answer = re.sub(r"</?T\d+>", '', ref_answer)
        assert len(pred_answer) and len(ref_answer)
        all_pred_answers[sample['sample_id']] = [pred_answer.replace("\n", "")]
        all_gt_answers[sample['sample_id']] = [ref_answer]
        id2idx[sample['sample_id']] = len(all_pred_answers.items()) - 1

    # Calculate BLEU scores
    bleu4_score, bleu4_scores = bleu4_scorer.compute_score(all_gt_answers, all_pred_answers)
    # Calculate METEOR scores
    meteor_score, meteor_scores = meteor_scorer.compute_score(all_gt_answers, all_pred_answers)
    # Calculate ROUGE scores, focusing on ROUGE-L
    rouge_l_score, rouge_l_scores = rouge_scorer.compute_score(all_gt_answers, all_pred_answers)
    # Calculate CIDER scores
    cider_score, cider_scores = cider_scorer.compute_score(all_gt_answers, all_pred_answers)
    # Calculate Similarity scores
    similarity_score, similarity_scores = similarity_scorer.compute_score(all_gt_answers, all_pred_answers)

    eval_results = []
    # Keep track of scores per category and overall score
    category_scores = defaultdict(list)
    overall_scores = []
    category_meteor = defaultdict(list)
    overall_meteor = []
    num_failed = 0
    for idx, sample in enumerate(tqdm(predictions)):

        label = labels[idx]
        for item in labels:
            if item['sample_id'] == sample['sample_id']:
                label = item

        assert sample['sample_id'] == label['sample_id']
        pred_answer = sample['A']
        ref_answer = label['A']
        category = sample['category']
        question = sample['Q']

        # skip when questions
        if re.search(r"<T\d+></T\d+>", ref_answer):
            continue
        # remove special tokens
        ref_answer = re.sub(r"</?T\d+>", '', ref_answer)

        # GPT-4 request
        try:
            response = get_response_gpt4(client, question, ref_answer, pred_answer)
            score, rationale = parse_response(response)
        except:
            num_failed += 1
            continue

        idx_ans = id2idx[sample['sample_id']]
        eval_results.append({
            "sample_id": sample['sample_id'],
            "question": question,
            "ref_answer": ref_answer,
            "pred_answer": pred_answer,
            "GPT-score": score,
            "rationale": rationale,
            "BLEU-4": bleu4_scores[3][idx_ans],
            "METEOR": meteor_scores[idx_ans],
            "ROUGE-L": rouge_l_scores[idx_ans],
            "CIDEr": cider_scores[idx_ans],
            "Similarity": similarity_scores[idx_ans],
        })
        category_scores[category].append(score)
        overall_scores.append(score)

    # Calculate average GPT-4 scores
    category_averages = {category: np.mean(scores) for category, scores in category_scores.items()}
    overall_average = np.mean(overall_scores)
    print(f"Category GPT-4o score: {category_averages}")
    print(f"Overall GPT-4o score: {overall_average}")

    print("=" * 60)
    print(f"BLEU-4: {bleu4_score[3]*100:.1f}")
    print(f"METEOR: {meteor_score*100:.1f}")
    print(f"ROUGE-L: {rouge_l_score*100:.1f}")
    print(f"CIDEr: {cider_score*100:.1f}")
    print(f"Similarity: {similarity_score*100:.1f}")

    print(f"#(Failed): {num_failed}")

    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(eval_results, f, indent=2)
