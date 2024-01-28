#!/usr/bin/env python

import srt
import argparse
from openai import OpenAI, AzureOpenAI
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


DEPLOYMENT_ID = os.getenv("AZURE_DEPLOYMENT_ID")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if DEPLOYMENT_ID and AZURE_ENDPOINT:
    client = AzureOpenAI(azure_deployment=DEPLOYMENT_ID, azure_endpoint=AZURE_ENDPOINT, api_key=OPENAI_API_KEY, api_version="2023-07-01-preview")
    print("using azure")
else:
    client = OpenAI(api_key=OPENAI_API_KEY)


BATCHSIZE = 50
LANG = "English"
MODEL = "gpt-3.5-turbo"
VERBOSE = False

def makeprompt():
    global prompt
    prompt = f"""You are a professional translator.
Translate the text below line by line into {LANG}, do not add any content on your own, and aside from translating, do not produce any other text, you will make the most accurate and authentic to the source translation possible.

these are subtitles, meaning each elements are related and in order, you can use this context to make a better translation with the original number of lines.
Please only return an JSON Array of trasnlated object with `index` and `content`, one for each line of the input, where the `content` field has the translated text.

For example, if the input is:
[{{"index": 1, "content": "Hello, world!"}}, {{ "index": 2, "content": "How are you?"}}]
the output should be like:
[{{"index": 1, "content": "Bonjour, monde!"}}, {{ "index": 2, "content": "Comment allez-vous?"}}]
"""

def makebatch(chunk):
    return [{'index': x.index, 'content': x.content} for x in chunk]

def translate_batch(batch):
    blen = len(batch)
    tbatch = []
    batch = json.dumps(batch, ensure_ascii=False)
    print(f"made batch {batch}")
    lendiff = 1
    while lendiff != 0:
        try:
            completion = client.chat.completions.create(model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": batch}
            ])
            tbatch_wip = json.loads(completion.choices[0].message.content)
            tbatch = parse_response(tbatch_wip)
        except Exception as e:
            if VERBOSE:
                print(e)
            lendiff = 1
        else:
            print(f"[debug] translated batch {tbatch}")
            lendiff = len(tbatch) - blen
    return tbatch

def parse_response(response):
    if "translation" in response:
        return response["translation"]
    elif "translations" in response:
        return response["translations"]
    elif "subtitles" in response:
        return response["subtitles"]
    elif "translatedTexts" in response:
        return response["translatedTexts"]
    else:
        return response

def translate_file(subs, num_workers: int):
    total_batch = (len(subs) + BATCHSIZE - 1) // BATCHSIZE
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {executor.submit(translate_batch, makebatch(subs[i:i+BATCHSIZE])): i for i in range(0, len(subs), BATCHSIZE)}
        
        for future in as_completed(future_to_batch):
            start_index = future_to_batch[future]
            try:
                translated_batch = future.result()
                for j, n in enumerate(translated_batch):
                    subs[start_index + j].content = n["content"]
            except Exception as e:
                if VERBOSE:
                    print(f"Error processing batch starting at index {start_index}: {e}")

def get_translated_filename(filepath):
    root, ext = os.path.splitext(os.path.basename(filepath))
    return f"{root}_{LANG}{ext}"

def main(files: str, language: str ="English", batch_size: int=50, model: str="gpt-3.5-turbo", verbose=False, num_workers: int = 5):
    global LANG, BATCHSIZE, MODEL, VERBOSE
    LANG = language
    BATCHSIZE = batch_size
    MODEL = model
    VERBOSE = verbose
    makeprompt()

    if not files:
        print("No files found matching the pattern.")
        return

    for filename in files:
        print(filename)
        sub = open(filename).read()
        subs = list(srt.parse(sub))

        translate_file(subs, num_workers=num_workers)
        output = srt.compose(subs)

        with open(get_translated_filename(filename), "w") as handle:
            handle.write(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate srt files")
    parser.add_argument("files", help="File pattern to match", nargs="+")
    parser.add_argument("-l", "--language", help="Specify the language", default="English", type=str)
    parser.add_argument("-b", "--batch_size", help="Specify the batch size", default=50, type=int)
    parser.add_argument("-m", "--model", help="openai's model to use", default="gpt-3.5-turbo", type=str)
    parser.add_argument("-v", "--verbose", help="display errors", action="store_true")
    parser.add_argument("-w", "--workers", help="number of workers for processing long file concurrently", default=5, type=int)

    args = parser.parse_args()

    main(args.files, args.language, args.batch_size, args.model, args.verbose, num_workers=args.workers)