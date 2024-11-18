import os
import sys
import json
import pandas as pd
import argparse

sys.path.append('/home/giuseppe/llms_as_judge/scripts')
os.chdir('/home/giuseppe/llms_as_judge/scripts')

import protocols


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action = "store", dest = 'model')
    parser.add_argument('--num_instances', action = "store", dest = 'sample_size', default = 100)
    args = parser.parse_args()

    data = pd.read_csv(f'/home/giuseppe/benchmarks/codereval/codereval4cs/input/codereval_sorted.csv')

    # generator_model = "codellama/CodeLlama-7b-Instruct-hf"
    # generator_model = "codellama/CodeLlama-13b-Instruct-hf"
    # generator_model = "codellama/CodeLlama-34b-Instruct-hf"

    # generator_model = "gpt-3.5-turbo"
    # generator_model = "gpt-4-turbo"

    generator_model = args.model
    sample_size = args.sample_size

    prompts = []
    output = []
    
    model_name = generator_model.split('/')[1] if 'gpt' not in generator_model else generator_model
    print(f'{model_name} generating summaries...\n')
    for i in range(data.shape[0])[:sample_size]:
        print(f'\rGenerating summary: {i+1}/{sample_size}.', end = '')
        method = data.target.iloc[i]
        # Pretend that you are an experienced Java programmer. 
        # Generate a short documentation (# Description) describing what the following Java method (# Java method) does:
        # prompt = f"""Generate a short documentation (# Description) describing what the following Java method (# Java method) does:
        prompt = f"""Pretend that you are an experienced Java programmer. Generate a short docstring (# Docstring) for the following Java method (# Java method):

# Java method
{method}

# Docstring:
"""
        if 'gpt' in model_name:
            model_output = protocols.ask_chatgpt(prompt = prompt, gpt_version = generator_model, language = 'Java')
        else:
            model_output = protocols.call_huggingface_model(model = generator_model, prompt = prompt, max_new_tokens = 512, task = 'SUMMARY GENERATION')
        prompts.append(prompt)
        output.append(model_output)
    
    data_d = data.head(sample_size)
    
    data_d['prompt'] = prompts
    data_d['generated_by'] = generator_model
    data_d[f'{model_name}_output'] = output

    # data_d.to_csv(f'/home/giuseppe/benchmarks/codereval/codereval4cs/summaries/{model_name}.csv', index = False)
