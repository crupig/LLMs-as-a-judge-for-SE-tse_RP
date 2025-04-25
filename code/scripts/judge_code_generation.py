import json
import os
import protocols
import pandas as pd

INPUT_PATH = '../data/input'
INPUT_FILE = 'CoderEval4Java.json'

BATCH_SIZE = 10
START_FROM_BATCH = 0

IDS_TO_DISCARD = json.load(open('../constants/constants_json.json'))["Ids with Unreliable Test"]

if __name__ == '__main__':

    #import java methods
    instances = json.load(open(os.path.join(INPUT_PATH, INPUT_FILE)))['RECORDS']
    instances = [jm for jm in instances if jm['_id'] not in IDS_TO_DISCARD]
    
    judge_model = "deepseek-ai/deepseek-coder-1.3b-instruct"
    # judge_model = "deepseek-ai/deepseek-coder-6.7b-instruct"
    # judge_model = "deepseek-ai/deepseek-coder-33b-instruct"
    # judge_model = "codellama/CodeLlama-7b-Instruct-hf"
    # judge_model = "codellama/CodeLlama-13b-Instruct-hf"
    # judge_model = "codellama/CodeLlama-34b-Instruct-hf"
    # judge_model = "gpt-3.5-turbo"
    # judge_model = "gpt-4-turbo"

    all_models = [
        "deepseek-coder-1.3b-instruct",
        "deepseek-coder-6.7b-instruct", 
        "deepseek-coder-33b-instruct",
        "CodeLlama-7b-Instruct-hf",
        "CodeLlama-13b-Instruct-hf",
        "CodeLlama-34b-Instruct-hf",
        "gpt-3.5-turbo",
        "gpt-4-turbo"
    ]

    language = INPUT_FILE.split('4')[1].split('.')[0].lower()
    output_path = f'../data/results/cg_judgement/{language}'

    batches = list(protocols.divide_into_batches(data = instances, batch_size = BATCH_SIZE))
    print(f'Testing {judge_model} as judge on {language.upper()}.')
    
    overall_df = pd.DataFrame()
    for index, batch in enumerate(batches[START_FROM_BATCH:]):
        batch_idx = index + START_FROM_BATCH
        batch_ids = [i['_id'] for i in batch]
        batch_target = [i['code'] for i in batch]
        batch_df = pd.DataFrame({'id' : batch_ids, 'target' : batch_target})

        for j, model in enumerate(all_models):
            print(f'\rBatch: {batch_idx + 1}/{len(batches)}, Model: {j + 1}/{len(all_models)}', end = "")

            ids = []
            code = []
            output = []
            predictions = protocols.extract_predictions(os.path.join(f'../data/results/code_generation/{language}', f'{model}.jsonl'), batch_ids)

            model_code = {}
            model_output = {}
            model_prompt = {}

            for id, candidate in zip(predictions.method_id, predictions.code):
                description = [c['human_label'] for c in batch if c['_id'] == id][0]
                targetMethod = [c['code'] for c in batch if c['_id'] == id][0]
                first_line = targetMethod.split('\n')[0].strip()

                if language == 'python' and len(first_line) > 0 and first_line[0] == '@':
                    signature = '\n'.join([line.strip() for line in targetMethod.split('\n')[:2]])
                elif language == 'python' and len(first_line) > 0 and first_line.split(' ')[0] == 'import':
                    signature = targetMethod.split('\n')[1].strip()
                else:
                    signature = targetMethod.split('\n')[0].strip()
                            
                prompt = f"""You will be provided with the description ("Description") and the signature ("Signature") of a {language} function to implement. You will also see a candidate implementation ("Candidate"). Your role is to evaluate the correctness of the Candidate, providing as output a rating (# Rating) and a rationale (# Rationale). Rate the Candidate on a scale from 0 to 1:

0. **Wrong Implementation**: The implementation does not correctly implement the described function.
1. **Correct Implementation**: The implementation correctly implements the described function.

# Description
{description}

# Signature
{signature}

# Candidate
{candidate}"""
                model_code.setdefault(model, [])
                model_output.setdefault(model, [])
                model_prompt.setdefault(model, [])
                
                if 'gpt-' in judge_model:
                    output = protocols.ask_chatgpt(prompt, judge_model, language)
                else:
                    output = protocols.call_model(model = judge_model, prompt = prompt, max_new_tokens = 250, task = 'JUDGEMENT')
                model_code[model].append(candidate)
                model_output[model].append(output)
                model_prompt[model].append(prompt)
            
            to_concat = pd.DataFrame({f'{model}_prompt' : list(model_prompt.values())[0], f'{model}_prediction' : list(model_code.values())[0], f'{model}_judgement' : list(model_output.values())[0]})
            batch_df = pd.concat([batch_df, to_concat], axis = 1)

        name = judge_model.split('/')[-1]
        
        overall_df = pd.concat([overall_df, batch_df])
        overall_df.to_csv(os.path.join(output_path, f'{name}.csv'), index = False)

    print()