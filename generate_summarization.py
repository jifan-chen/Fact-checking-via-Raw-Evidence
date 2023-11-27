import json
import re
import nltk.tokenize
import pandas as pd
import argparse
import openai
import os
from tqdm import tqdm
from datetime import datetime, timedelta
from nltk.tokenize import word_tokenize
from collections import OrderedDict
from typing import List, Tuple
import models.BM25_retriever as BM25


openai.api_key = os.getenv("OPENAI_API_KEY")
OFFSET = 150
unit2doc_name = {}
unit2span = {}
doc_name2doc = {}
REGEX = "(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?" \
            "|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?" \
            "|Dec(?:ember)?)\s+(\d{1,2}),?\s+(\d{4})"

MONTH_MAPPING = {"Jan": "01",
                 "January": "01",
                 "Feb": "02",
                 "February": "02",
                 "Mar": "03",
                 "March": "03",
                 "Apr": "04",
                 "April": "04",
                 "May": "05",
                 "Jun": "06",
                 "June": "06",
                 "Jul": "07",
                 "July": "07",
                 "Aug": "08",
                 "August": "08",
                 "Sep": "09",
                 "September": "09",
                 "Oct": "10",
                 "October": "10",
                 "Nov": "11",
                 "November": "11",
                 "Dec": "12",
                 "December": "12"
                 }


def extract_claim_date(claim_context, window=None):
    res = re.findall(REGEX, claim_context)
    if res:
        month, day, year = res[0]
        if int(day) < 10:
            day = '0' + day
        date_str = "{}-{}-{}".format(year, MONTH_MAPPING[month], day)
        if window:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            # add one month to the date
            new_date = date_obj + timedelta(days=window)
            # format the new date as "YYYY-MM-DD"
            new_date_str = new_date.strftime("%Y-%m-%d")
            return new_date_str
        else:
            return date_str
    else:
        return None


def filter_corpus_by_timestamp(docs, doc_links, timestamps, when_where, window=None):
    timestamp = extract_claim_date(when_where, window=window)
    selected = []
    for i in range(len(timestamps)):
        if timestamps[i]:
            date1 = datetime.strptime(timestamps[i], '%Y-%m-%d')
            date2 = datetime.strptime(timestamp, '%Y-%m-%d')
            if date1 < date2:
                selected.append(i)
        else:
            selected.append(i)
    docs = [docs[i] for i in selected]
    doc_links = [doc_links[i] for i in selected]
    return docs, doc_links


def map_link2name(pages_info):
    link2name = {}
    for info in pages_info:
        link2name[info['page_url']] = info['page_name']
    return link2name


def generate_one_summ_via_GPT(extracted_docs, claim, use_ChatGPT):
    prompt = ''
    for i, doc_name in enumerate(extracted_docs.keys()):
        content = extracted_docs[doc_name]
        prompt += f"Document title: {doc_name}\n"
        prompt += f"Content: {content} \n"
        prompt += f"Claim: {claim} \n"
    prompt += f'As a professional fact-checker, your task is to summarize the above document in 6-8 sentences to help fact-check the above claim. Your response should provide a clear and concise summary of the relevant information contained in the document while not make any judgment about the claim. You should focus on identifying key facts and details that are relevant to assessing the accuracy of the claim, while ignoring the information that is unrelated. \nSummarization:'
    if not use_ChatGPT:
        ENGINE = 'text-davinci-003'
        res = openai.Completion.create(engine=ENGINE, prompt=prompt,
                                       temperature=0.75, max_tokens=512,
                                       stop=["Claim"])
        res = res['choices'][0]["text"].strip()
    else:
        ENGINE = 'gpt-3.5-turbo'
        res = openai.ChatCompletion.create(
            model=ENGINE,
            messages=[{"role": "user", "content": f"{prompt}"}])
        res = res['choices'][0]["message"]["content"].strip()
    return res, prompt


def generate_multi_summs_via_GPT(extracted_docs, claim, use_ChatGPT):
    full_prompt = ""
    full_result = ""
    for i, doc_name in enumerate(extracted_docs.keys()):
        prompt = ""
        content = extracted_docs[doc_name]
        prompt += f"Document title: {doc_name}\n"
        prompt += f"Content: {content} \n"
        full_prompt += prompt + '\n'
        prompt += f"Claim: {claim} \n"
        prompt += f'As a professional fact-checker, your task is to summarize the above document in 1-2 sentences to help fact-check the above claim. Your response should provide a clear and concise summary of the relevant information contained in the document while not make any judgment about the claim. You should focus on identifying key facts and details that are relevant to assessing the accuracy of the claim, while ignoring the information that is unrelated. \nSummarization:'

        if not use_ChatGPT:
            ENGINE = 'text-davinci-003'
            res = openai.Completion.create(engine=ENGINE, prompt=prompt,
                                           temperature=0.75, max_tokens=512,
                                           stop=["Claim"])
            res = f"Document {i}: " + res['choices'][0]["text"].strip()
            full_result = full_result + "\n" + res
        else:
            ENGINE = 'gpt-3.5-turbo'
            res = openai.ChatCompletion.create(
                model=ENGINE,
                messages=[{"role": "user", "content": f"{prompt}"}])
            res = f"Document {i}: " + res['choices'][0]["message"]["content"].strip()
            full_result = full_result + "\n" + res
    return full_result, full_prompt


def generate_multi_summs_via_GPT_few_shot(extracted_docs, claim, use_ChatGPT):
    few_shot_prompt = open('./few-shot-prompt.txt', 'r').read()
    full_prompt = ""
    full_result = ""
    for i, doc_name in enumerate(extracted_docs.keys()):
        prompt = few_shot_prompt + '\n'
        content = extracted_docs[doc_name]
        prompt += f"Document: {doc_name}\n"
        prompt += f"Content: {content} \n\n"
        prompt += f'Suppose you are assisting a fact-checker to fact-check the claim:\n {claim} \n Summarize the relevant information from the document in 1-2 sentences. Your response should provide a clear and concise summary of the relevant information contained in the document. Do not include a judgment about the claim and do not repeat any information from the claim that is not supported by the document.'
        full_prompt += f"Document title: {doc_name}\n"
        full_prompt += f"Content: {content} \n\n"
        if not use_ChatGPT:
            ENGINE = 'text-davinci-003'
            res = openai.Completion.create(engine=ENGINE, prompt=prompt,
                                           temperature=0.75, max_tokens=256,
                                           stop=["Claim"])
            res = f"Document {i}: " + res['choices'][0]["text"].strip()
            full_result = full_result + "\n" + res
        else:
            ENGINE = 'gpt-3.5-turbo'
            res = openai.ChatCompletion.create(
                model=ENGINE,
                messages=[{"role": "user", "content": f"{prompt}"}])
            res = f"Document {i}: " + res['choices'][0]["message"]["content"].strip()
            full_result = full_result + "\n" + res
    return full_result, full_prompt


def merge_spans(spans: List[Tuple[int, int]], offset: int = 200):
    # Sort the spans by their start position
    spans = [(max(0, s[0] - offset), s[1] + offset) for s in spans]
    spans.sort(key=lambda x: x[0])
    merged_spans = []
    current_span = spans[0]
    for span in spans[1:]:
        # If the current span and the next span overlap, merge them
        if span[0] <= current_span[1]:
            current_span = (current_span[0], max(span[1], current_span[1]))
        # If the current span and the next span don't overlap, add the current span to the list of merged spans
        else:
            merged_spans.append(current_span)
            current_span = span
    # Add the last span to the list of merged spans
    merged_spans.append(current_span)

    return merged_spans


def filter_corpus_by_politifact(docs, doc_links):
    selected_ids = []
    for i, link in enumerate(doc_links):
        if 'politifact' in link:
            selected_ids.append(i)
    selected_docs = [docs[i] for i in selected_ids]
    selected_doc_links = [doc_links[i] for i in selected_ids]
    return selected_docs, selected_doc_links


def get_query(df_row, args):
    if args.use_annotation:
        qs1 = df_row['annotations'][0]['questions']
        qs2 = df_row['annotations'][1]['questions']
        questions = qs1 if len(qs1) > len(qs2) else qs2
    elif args.use_claim:
        questions = [df_row['claim']]
    elif args.use_justification:
        questions = [df_row['justification']]
    else:
        questions = df_row['qg-output']
    return questions


def feed_claim_with_context(df_row):
    return f"{df_row['person']} {df_row['venue']} {df_row['claim']}"


def main(args):
    total_num_docs = []
    total_num_words = []
    corpus = json.load(open(args.corpus_path))
    df = pd.read_json(args.input_path, lines=True, dtype={'example_id': 'string'})
    df = df.drop(columns=['summary_timestamp', 'summarization_prompt'])
    if 'summary_timestamp' not in df.columns:
        df['summary_timestamp'] = ""
    if 'summarization_prompt' not in df.columns:
        df["summarization_prompt"] = ""
    for i, row in tqdm(df.iterrows()):
        if row['summary_timestamp']:
            continue
        example_id = str(row['example_id'])
        docs = corpus[example_id]['text']
        doc_links = corpus[example_id]['links']
        timestamps = corpus[example_id]['timestamp']
        if args.filter_politifact:
            docs, doc_links = filter_corpus_by_politifact(
                docs, doc_links)
        # Bing search use the page created time as timestamp, which could be modified later
        # We do another round of filtering to make sure the latest modification of page timestamp is before the claim
        if args.time_restricted:
            print('filtering by timestamp.......')
            docs, doc_links = filter_corpus_by_timestamp(
                docs, doc_links, timestamps, row['venue'],
                window=args.time_window
            )

        restricted_pages = row['search_results_timestamp']['pages_info'] if \
            'search_results_timestamp' and "search_results_timestamp" in row else []
        link2name = map_link2name(restricted_pages)
        doc_names = [link2name[link] for link in doc_links]
        questions = get_query(row, args)
        text_units = []

        for doc, name in zip(docs, doc_names):
            if args.text_unit == "doc":
                tokenized_doc = word_tokenize(" ".join(doc))
                doc_string = " ".join(tokenized_doc)
                text_units.append(tokenized_doc)
                unit2doc_name[doc_string] = name
            else:
                tokenized_doc = word_tokenize(" ".join(doc))
                doc_name2doc[name] = tokenized_doc
                for start in range(
                        0, len(tokenized_doc) - args.window_size,
                        args.stride):
                    end = start + args.window_size
                    unit = tokenized_doc[start:end]
                    text_units.append(unit)
                    unit2doc_name[' '.join(unit)] = name
                    unit2span[' '.join(unit)] = (start, end)

        # The first-stage retrieval using annotated questions may result in empty docs
        # in such cases, use the claim itself as dummy
        if not len(text_units):
            df.at[i, 'summary_timestamp'] = row['claim']
            df.at[i, 'summarization_prompt'] = ""
            continue
        retriever = BM25.BM25Retriever(text_units)
        tokenized_questions = [word_tokenize(q) for q in questions]
        retrieve_results = []
        units_so_far = set()
        for tq in tokenized_questions:
            units, scores = retriever.get_top_n_doc(tq, num=args.num)
            for u, s in zip(units, scores):
                u = " ".join(u)
                if u not in units_so_far:
                    units_so_far.add(u)
                    retrieve_results.append((u, s))
        retrieve_results = sorted(retrieve_results, key=lambda x: x[1], reverse=True)[:args.topk_units]

        doc_name2spans = OrderedDict()
        for res in retrieve_results:
            text_unit = res[0]
            name = unit2doc_name[text_unit]
            span = unit2span[text_unit]
            if name not in doc_name2spans:
                doc_name2spans[name] = [span]
            else:
                doc_name2spans[name].append(span)
        doc_name2span_text = {}
        for name, spans in list(doc_name2spans.items())[:args.topk_docs]:
            doc_name2spans[name] = merge_spans(spans, offset=OFFSET)
            content = ""
            for s in doc_name2spans[name]:
                content += " ".join(doc_name2doc[name][s[0]: s[1]])
            doc_name2span_text[name] = content
        claim = feed_claim_with_context(row)
        try:
            if args.summ_type == "one_summ":
                rationale, prompt = generate_one_summ_via_GPT(
                    doc_name2span_text, claim=claim, use_ChatGPT=args.use_ChatGPT)
            elif args.summ_type == "multi_summs" and args.use_fewshot:
                rationale, prompt = generate_multi_summs_via_GPT_few_shot(
                    doc_name2span_text, claim=claim, use_ChatGPT=args.use_ChatGPT)
            elif args.summ_type == "multi_summs" and not args.use_fewshot:
                rationale, prompt = generate_multi_summs_via_GPT(
                    doc_name2span_text, claim=claim, use_ChatGPT=args.use_ChatGPT)
            else:
                num_words = []
                for doc_name, text in doc_name2span_text.items():
                    num_words.append(len(nltk.tokenize.word_tokenize(doc_name + " " + text)))
                total_num_docs.append(len(num_words))
                total_num_words.append(sum(num_words))
                raise ValueError("Unknown summ_type")
            df.at[i, 'summary_timestamp'] = rationale
            df.at[i, 'summarization_prompt'] = prompt
        except Exception as e:
            print(e)
            print("current i:", i)
            df.to_json(args.output_path, orient='records', lines=True)
            df[['summarization_prompt', 'summary_timestamp', 'claim', 'justification', 'label']].to_csv(
                args.output_path + ".csv", index=False)

    df.to_json(args.output_path, orient='records', lines=True)
    df[['summarization_prompt', 'summary_timestamp', 'claim', 'justification', 'label']].to_csv(
        args.output_path + ".csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None, help="path to the input file containing bing search results")
    parser.add_argument('--corpus_path', type=str, default=None, help="path to the corpus file containing scraped web docs")
    parser.add_argument('--output_path', type=str, default=None, help="path to the output file")
    parser.add_argument('--num', type=int, default=5,
                        help="number of retrieved text units per question")
    parser.add_argument('--window_size', type=int, default=30, help="chunk size for BM25 retrieval")
    parser.add_argument('--topk_units', type=int, default=10, help="number of retrieved text units per question")
    parser.add_argument('--topk_docs', type=int, default=5, help="number of retrieved docs ranked by BM25 score")
    parser.add_argument('--stride', type=int, default=15, help="stride of document chunking")
    parser.add_argument('--use_claim', type=int, default=0, help="whether to use claim as input")
    parser.add_argument('--use_annotation', type=int, default=0, help="whether to use annotated questions")
    parser.add_argument('--use_justification', type=int, default=0, help="whether to use justification as input")
    parser.add_argument('--text_unit', type=str, default="span",
                        help="choose from paragraph and span")
    parser.add_argument('--time_restricted', type=int, default=1, help="whether to use time restricted search results")
    parser.add_argument('--time_window', type=int, default=1, help="time window in days")
    parser.add_argument('--summ_type', type=str, default="multi_summs")
    parser.add_argument('--use_ChatGPT', type=int, default=0, help="whether to use ChatGPT for summarization")
    parser.add_argument('--use_fewshot', type=int, default=1, help="whether to use few-shot prompting for summarization")
    parser.add_argument('--filter_politifact', type=int, default=0, help="whether to filter out politifact docs")
    main(parser.parse_args())
