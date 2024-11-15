import os
import protocols
import pandas as pd
import argparse

INPUT_PATH = '../data/input'

MA_BATCHES = ['Batch1', 'Batch2', 'Batch3', 'Batch4', 'Batch5']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action = "store", dest = 'model')
    parser.add_argument('--max_tokens', action = "store", dest = 'max_new_tokens', default = 512, type = int)
    args = parser.parse_args()

    judge_model = args.model

    # models = [
    #     "deepseek-ai/deepseek-coder-1.3b-instruct",
    #     "deepseek-ai/deepseek-coder-6.7b-instruct", 
    #     "deepseek-ai/deepseek-coder-33b-instruct",
    #     "codellama/CodeLlama-7b-Instruct-hf",
    #     "codellama/CodeLlama-13b-Instruct-hf",
    #     "codellama/CodeLlama-34b-Instruct-hf",
    #     "gpt-3.5-turbo",
    #     "gpt-4-turbo"
    # ]

    output_path = '../data/results/tse/cs_judgement_codereval'
    print(f'Evaluating {judge_model} as judge on code summarization.\n')
    
    for ibatch, batch in enumerate(MA_BATCHES):
        data = pd.read_csv(os.path.join(INPUT_PATH, f'{batch}_cs_manualanalysis.csv'))
        data = data[['target_id', 'target', 'generated_by', 'summary', 'summary_postprocessed']]
        data['batch'] = batch

        prompts = []
        model_outputs = []
        for iinstance in range(data.shape[0]):

            print(f'\rBatch: {ibatch + 1}/{len(MA_BATCHES)}, Instance: {iinstance + 1:03d}/{data.shape[0]}', end = "")

            method = data['target'].iloc[iinstance]
            comment = data['summary_postprocessed'].iloc[iinstance]

            prompt = f"""You will be provided with a Java function ("Function") and a textual summary of it ("Comment"). The goal of the Comment is to document the functionality implemented in the Function. Your role is to evaluate the Comment across three criteria, providing as output for each of them a rating (# Rating) and a rationale (# Rationale) as described in the following.

# Evaluation Criteria
* Content adequacy: the extent to which the comment summarizes all information that can be inferred from the source code.

* Conciseness: the extent to which the comment contains unnecessary information.

* Fluency & Understandability: the extent to which the comment is easy to read and understand.

For each criterion, provide a score on a scale from 1 to 5:

1. Very poor
2. Poor
3. Fair
4. Good
5. Very good

# Function
{method}

# Comment
{comment}
#"""
            
            if 'gpt-' in judge_model:
                output = protocols.ask_chatgpt(prompt = prompt, gpt_version = judge_model, language = 'java')
            else:
                output = protocols.call_model(model = judge_model, prompt = prompt, max_new_tokens = args.max_new_tokens, task = 'JUDGEMENT')
            
            prompts.append(prompt)
            model_outputs.append(output)

        data['prompt'] = prompts
        data['model_output'] = model_outputs
        model_name = judge_model.split('/')[-1]
        data.to_csv(os.path.join(output_path, f'{model_name}_{batch}.csv'), index = False)

    print()