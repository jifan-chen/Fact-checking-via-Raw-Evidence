import numpy as np
from rouge_score import rouge_scorer
from collections import OrderedDict


SEP_TK = "[SEP]"
ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)


def feed_claim_with_context(examples):
    input_text = []
    for c, p, v in zip(
        examples['claim'],
        examples['person'],
        examples['venue']
    ):
        input_text.append(
            f"{p} {v} {c}"
        )

    return input_text


def concat_claim_retrieved_sents(input_text, examples, topk_sents=10, topk_docs=2):
    # merge all sentences retrieved by each claim according to the retrieval
    # scores
    res = []
    evd_set = set()
    questions = examples['qg-output']
    for raw_claim, claim, evd_sents, evd_scores, mapping_names in zip(
            examples['claim'],
            input_text,
            examples['topk-text-units'],
            examples['topk-scores'],
            examples['topk-doc-names'],
    ):
        all_sents = sum(evd_sents, [])
        all_scores = sum(evd_scores, [])
        all_names = sum(mapping_names, [])
        dedup_sents = []
        dedup_scores = []
        for sent, score, doc_name in zip(all_sents, all_scores, all_names):
            claim_rouge_score = ROUGE_SCORER.score(raw_claim, sent)
            reference_rouge_score = ROUGE_SCORER.score(sent, raw_claim)
            if sent not in evd_set and claim_rouge_score['rouge1'][1] < 0.8 \
                    and reference_rouge_score['rouge1'][1] < 0.8:
                evd_set.add(sent)
                dedup_sents.append((sent, doc_name))
                dedup_scores.append(score)
        selected_ids = np.argsort(dedup_scores)[::-1][:topk_sents]
        selected_sents = [dedup_sents[i] for i in selected_ids]

        name2sents = OrderedDict()
        for item in selected_sents:
            if item[1] in name2sents:
                name2sents[item[1]].append(item[0])
            else:
                name2sents[item[1]] = [item[0]]

        evidence = ""
        for idx, k in enumerate(list(name2sents.keys())[:topk_docs]):
            sents = name2sents[k]
            concat_sents = " | ".join(sents)
            evidence += f"Doc{idx}: {k} {SEP_TK} Content: {concat_sents} "
        inputs = f'{claim} {SEP_TK} {evidence}'
        res.append(inputs)
    return res


def concat_claim_gpt_rationale(input_text, examples):
    res = []
    for claim, rationale in zip(
            input_text,
            examples['summary']
    ):
        res.append(
            f'{claim} {SEP_TK} {rationale}'
        )
    return res


def concat_claim_justification(input_text, examples):
    res = []
    for claim, just in zip(
        input_text,
        examples['justification']
    ):
        res.append(
            f'{claim} {SEP_TK} {just}'
        )
    return res


def concat_claim_qa_pairs(input_text, examples):
    res = []
    for claim, annot in zip(
        input_text,
        examples['annotations']
    ):
        if len(annot[0]['questions']) > len(annot[1]['questions']):
            qs = annot[0]['questions']
            ans = annot[0]['answers']
        else:
            qs = annot[1]['questions']
            ans = annot[1]['answers']
        qas = [f'{a}' for q, a in zip(qs, ans)]
        qas = " | ".join(qas)
        inputs = f'{claim} {SEP_TK} {qas}'
        res.append(inputs)
    return res


def concat_claim_questions(input_text, examples, generated_question=True):
    res = []
    if not generated_question:
        for claim, annot in zip(
            input_text,
            examples['annotations']
        ):
            if len(annot[0]['questions']) > len(annot[1]['questions']):
                questions = annot[0]['questions']
            else:
                questions = annot[1]['questions']
            questions = " | ".join(questions)
            inputs = f'{claim} {SEP_TK} {questions}'
            res.append(inputs)
    else:
        for claim, qs in zip(
            input_text,
            examples['qg-output']
        ):
            questions = " | ".join(qs)
            inputs = f'{claim} {SEP_TK} {questions}'
            res.append(inputs)
    return res
