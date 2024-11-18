import os
import json
import pandas as pd
import argparse
from pygments.lexers import JavaLexer

import protocols


INPUT_PATH = '../data/input'
INPUT_FILE = 'CoderEval4Java.json'

IDS_TO_DISCARD = json.load(open('../constants/constants_json.json'))["Ids with Unreliable Tests"]

if __name__ == '__main__':

    instances = json.load(open(os.path.join(INPUT_PATH, INPUT_FILE)))['RECORDS']
    instances = [jm for jm in instances if jm['_id'] not in IDS_TO_DISCARD]

    lexer = JavaLexer()
    _ids = []
    targets = []
    num_tokens = []

    for instance in instances:
        target_id = instance['_id']
        target = instance['code']
        target_num_tokens = len(list(lexer.get_tokens(target)))

        _ids.append(target_id)
        targets.append(target)
        num_tokens.append(target_num_tokens)

    
    data = pd.DataFrame({
        'id' : _ids,
        'target' : targets,
        'num_tokens' : num_tokens
    })

    data = data.sort_values(by = 'num_tokens', ascending = False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action = "store", dest = 'model')
    parser.add_argument('--num_instances', action = "store", dest = 'sample_size', default = 100)
    args = parser.parse_args()

    # list of models from which to choose a summary generator
    # models = [
    #     "codellama/CodeLlama-7b-Instruct-hf", 
    #     "codellama/CodeLlama-13b-Instruct-hf",
    #     "codellama/CodeLlama-34b-Instruct-hf",
    #     "gpt-3.5-turbo",
    #     "gpt-4-turbo"
    #     ]
    
    generator_model = args.model
    sample_size = args.sample_size

    prompts = []
    output = []
    
    model_name = generator_model.split('/')[1] if 'gpt' not in generator_model else generator_model
    print(f'{model_name} generating summaries...\n')
    for i in range(data.shape[0])[:sample_size]:
        print(f'\rGenerating summary: {i+1}/{sample_size}.', end = '')
        method = data.target.iloc[i]
        prompt = f"""Pretend that you are an experienced Java programmer. Generate a short docstring (# Docstring) for the following Java method (# Java method):

# Java method
{method}

# Docstring:
"""
        if 'gpt' in model_name:
            model_output = protocols.ask_chatgpt(prompt = prompt, gpt_version = generator_model, language = 'Java')
        else:
            model_output = protocols.call_model(model = generator_model, prompt = prompt, max_new_tokens = 512, task = 'SUMMARY GENERATION')
        prompts.append(prompt)
        output.append(model_output)
    
    data = data.head(sample_size)
    
    data['prompt'] = prompts
    data['generated_by'] = generator_model
    data[f'model_output'] = output

    data.to_csv(f'../data/results/cs/{model_name}_CCF.csv', index = False)
