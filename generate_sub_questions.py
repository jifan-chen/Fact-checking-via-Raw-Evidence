import argparse
import os
import re
import openai
import pandas as pd
from tqdm import tqdm

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

WH_MATCHES = ("why", "who", "which", "what", "where", "when", "how")

NUMBERS = ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")

# OpenAI Engine, feel free to change
ENGINE = 'text-davinci-003'
# The stop criteria for question generation, feel free to change
MAX_GPT_CALLS = 5
MAX_NUM_QUESTIONS = 10


# Format example for static prompt
def construct_static_examples():
    showcase_examples = '''Claim: Viral image stated on June 8, 2020 in post on Facebook: Cops in Norway: require 3 years of training, 4 people killed since 2002. Cops in Finland: require 2 years of training, 7 people killed since 2000. Cops in Iceland: require 2 years of training, 1 person killed since ever. Cops in the U.S.: require 21 weeks of training, 8,000+ people killed since 2001.

Suppose you are a fact-checker, generate several yes or no quesons to help me answer if this claim is true or false.

Quesons:
Does Norway require 3 years of training for cops?
Have Norwegian cops killed 4 people since the early 2000's?
Does Finland require only 2 years of training for police?
Have Finnish police killed 7 people since 2000?
Does Iceland only require 2 years of training for cops?
Have Iceland cops only killed 1 person ever?
Does the U.S. require only 21 weeks of training for cops?
Have U.S. cops killed more than 8,000 people since 2001?
Do experts associate only training me with police-related shoong fatalies?

Claim: Barry DuVal stated on September 25, 2015 in an interview: We're the only major oil-producing naon in the world with a self-imposed ban on exporng our crude oil to other naons.

Suppose you are a fact-checker, generate several yes or no quesons to help me answer if this claim is true or false.

Questions:
Is the U.S. the only major oil-producing naon to ban exports of crude oil?
Is the self-imposed ban on crude oil export of U.S a complete ban?

Claim: William Barr stated on September 2, 2020 in a CNN interview: We indicted someone in Texas, 1,700 ballots collected from people who could vote, he made them out and voted for the person he wanted to.

Suppose you are a fact-checker, generate several yes or no quesons to help me answer if this claim is true or false.

Questions:
Were 1700 mail-in ballots invesgated for fraud in Texas during the 2020 elecon?
Did the Justice Department indict someone in Texas for voter fraud?
Did widespread mail-in order fraud happen in Texas during the 2020 elecon?
Did voter disenfranchisement happen in Texas during the 2020 elecon?

Claim: {}
Suppose you are a fact-checker, generate several yes or no quesons to help me answer if this claim is true or false.

Questions:'''
    return showcase_examples

# Format context for prompt
def construct_context(person, venue, claim):
    return "{} {} {}".format(
        person,
        venue,
        claim
    )


# Format prompt for GPT3 call
def construct_prompt(showcase_examples, context):
    return showcase_examples.format(context)


# Generate sub-questions with GPT3 and apply filtering 
def generate_questions_by_gpt3(context, questions_by_gpt3):
    showcase_examples = construct_static_examples()
    prompt = construct_prompt(showcase_examples, context)
    res = openai.Completion.create(engine=ENGINE, prompt=prompt,
                                   temperature=0.7, max_tokens=1024,
                                   stop=["Claim"])
    res = res['choices'][0]["text"].strip()
    for q in res.splitlines():
        is_added = True
        q = q.strip()

        if len(q) != 0:
            # Remove quotation mark if there are any
            q = re.sub('"', '', q)
            # Remove question number if there are any
            if q.lower().startswith(NUMBERS):
                q = q[3:]

            # Do not add question that has wh-word or is duplicate
            if not q.isspace() and not any(x in q.lower() for x in WH_MATCHES):
                for q_alreay_in in questions_by_gpt3:
                    # Duplicate if exact string match
                    if q == q_alreay_in:
                        is_added = False
                        break
            else:
                is_added = False

            if is_added:
                questions_by_gpt3.append(q)
    return questions_by_gpt3


def main(args):
    df = pd.read_json(args.input_path, lines=True)

    # Add empty column if we don't have already
    if 'qg-output' not in df.columns:
        df["qg-output"] = ""
    start = 0 if not args.start else args.start
    end = len(df) if not args.end else args.end

    for i in tqdm(range(start, end)):
        if not df.iloc[i]['qg-output'] == "":
            print(f"question already generated at location {i}")
            continue
        try:
            context = construct_context(
                df.iloc[i]['person'], df.iloc[i]['venue'], df.iloc[i]['claim'])
            gpt3_called = 0
            questions_by_gpt3 = []
            while (len(questions_by_gpt3) < MAX_NUM_QUESTIONS
                   and gpt3_called < MAX_GPT_CALLS):
                questions_by_gpt3 = generate_questions_by_gpt3(
                    context, questions_by_gpt3
                )
                gpt3_called += 1

            df.at[i, 'qg-output'] = questions_by_gpt3
        except Exception as e:
            print("error caught", e)
            print('i=', i)

    df.to_json(args.output_path, orient='records', lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()
    main(args)
