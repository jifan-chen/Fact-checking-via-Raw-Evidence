import re
import argparse
import pandas as pd
import models.raw_evidence_retriever as retriever
from tqdm import tqdm
from typing import Dict, Set
from datetime import datetime, timedelta
from multiprocessing import Pool

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


def extract_claim_date(claim_context, time_offset):
    res = re.findall(REGEX, claim_context)
    if res:
        month, day, year = res[0]
        if int(day) < 10:
            day = '0' + day
        date_str = "{}-{}-{}".format(year, MONTH_MAPPING[month], day)
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        # add offset to the date so that we can experiment with different time constraints
        new_date = date_obj + timedelta(days=time_offset)
        # format the new date as "YYYY-MM-DD"
        new_date_str = new_date.strftime("%Y-%m-%d")
        return new_date_str
    else:
        return None


def update_results(
        accumulated_res: Dict,
        res: Dict,
        all_urls: Set,
        all_entity_names: Set
):
    for entity_info in res['entities_info']:
        if entity_info['name'] not in all_entity_names:
            accumulated_res['entities_info'].append(entity_info)
            all_entity_names.add(entity_info['name'])

    for page_info in res['pages_info']:
        if page_info['page_url'] not in all_urls:
            accumulated_res['pages_info'].append(page_info)
            all_urls.add(page_info['page_url'])


def process_chunk(chunk, chunk_idx, retriever, args):
    try:
        for i, row in tqdm(chunk.iterrows()):
            # skip if the row has already been processed with time constraint
            if args.use_time_stamp and row['search_results_timestamp']:
                print(f'row {i} in chunk {chunk_idx} has already been processed with time constraint')
                continue
            # skip if the row has already been processed without time constraint
            if not args.use_time_stamp and row['search_results']:
                print(f'row {i} in chunk {chunk_idx} has already been processed without time constraint')
                continue
            when_where = row['venue']
            timestamp = extract_claim_date(when_where, args.time_offset)
            if args.use_annotation:
                qs1 = row['annotations'][0]['questions']
                qs2 = row['annotations'][1]['questions']
                questions = qs1 if len(qs1) > len(qs2) else qs2
            elif args.use_claim:
                claim = f"{row['person']} {row['venue']} {row['claim']}"
                questions = [claim]
            else:
                questions = row['qg-output']
            questions = [q for q in questions if q.strip()]
            all_urls = set()
            all_entity_names = set()
            results = {
                'entities_info': [],
                'pages_info': []
            }
            for q in questions[:args.question_num]:
                if args.use_time_stamp:
                    res = retriever.get_results(q, timestamp)
                else:
                    res = retriever.get_results(q)
                update_results(results, res, all_urls, all_entity_names)
            if not args.use_time_stamp:
                chunk.at[i, 'search_results'] = results
            else:
                chunk.at[i, 'search_results_timestamp'] = results
            print(len(results['entities_info']), len(results['pages_info']))

    except Exception as e:
        print('error:', e)
        print(f'current index = {i}, chunk = {chunk_idx}')
        return chunk
    return chunk


def main(args):
    if 'jsonl' in args.input_url:
        df = pd.read_json(args.input_url, lines=True)
    else:
        url = args.url.replace('/edit#gid=', '/export?format=csv&gid=')
        df = pd.read_csv(url)
    # df = df.drop(columns=['search_results_timestamp'])
    # df = df.drop(columns=['search_results'])
    if 'search_results' not in df.columns:
        df["search_results"] = ""
    if 'search_results_timestamp' not in df.columns:
        df['search_results_timestamp'] = ""

    web_retriever = retriever.WebRetriever(
        engine='bing',
        answer_count=args.answer_count,
        sites_constrain=args.sites_constrain,
    )
    start = args.start
    end = min(len(df), args.end)
    chunks = [df[i:i + args.chunk_size] for i in range(start, end, args.chunk_size)]
    with Pool() as pool:
        process_arg = [(chunk, idx, web_retriever, args) for idx, chunk in enumerate(chunks)]
        results = pool.starmap(process_chunk, process_arg)
    # Separate the results into two lists
    # Merge the results back into a single DataFrame
    df_processed = pd.concat([r for r in results if r is not None])
    for i, row in df_processed.iterrows():
        if args.use_time_stamp and not row['search_results_timestamp']:
            print(f"error happens at row {i} when searching with timestamp")
        if not args.use_time_stamp and not row['search_results_timestamp']:
            print(f"error happens at row {i} when searching")
    df_processed.to_json(args.output_path, orient='records', lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_url', type=str, default=None,
                        help="url of the input file, could be a local jsonl or a google sheet")
    parser.add_argument('--output_path', type=str, default=None, help="path of the output file")
    parser.add_argument('--use_time_stamp', type=int, default=1, help="whether to use time stamp as search constraint")
    parser.add_argument('--sites_constrain', type=int, default=1,
                        help="whether to constrain the search to certain sites")
    parser.add_argument('--use_annotation', type=int, default=0, help="whether to use annotated questions")
    parser.add_argument('--use_claim', type=int, default=0, help="whether to use claim as question")
    parser.add_argument('--question_num', type=int, default=10, help="number of questions to use")
    parser.add_argument('--answer_count', type=int, default=10, help="number of answers to retrieve")
    parser.add_argument('--start', type=int, default=0, help="start index of the data to do retrieval")
    parser.add_argument('--end', type=int, default=1000, help="end index of the data to do retrieval")
    parser.add_argument('--chunk_size', type=int, default=50, help="size of the chunk")
    parser.add_argument('--time_offset', type=int, default=1, help="add an offest to the time at which the claim was made")
    main(parser.parse_args())
