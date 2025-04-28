import os
import ModelQuerier
import pandas as pd
import argparse
from tqdm import tqdm
from datetime import datetime

DEBUG_MODE = '../'

INPUT_PATH = f'{DEBUG_MODE}data/input/cs'
OUTPUT_ROOT = f'{DEBUG_MODE}data/results/tse/tse_maj/cs'

def divide_into_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


if __name__ == '__main__':
    now = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action = "store", dest = 'model')
    parser.add_argument(
        '--language',
        choices = ['java', 'python'],
        dest = 'language',
        help = "Choose one of the following values: 'java', 'python'."
        )
    parser.add_argument(
        '--judgment_type',
        choices = ['zeroshot', 'extended_instructions', 'stepbystep', 'stepbystep_extended_instructions'],
        dest = 'judgment_type',
        help = "Choose one of the following values: 'zeroshot', 'extended_instructions', 'stepbystep', 'stepbystep_extended_instructions'."
        )
    parser.add_argument(
        '--allocate',
        choices = ['yes', 'no'],
        dest = 'allocate',
        help = "Choose one of the following values: 'yes', 'no'.",
        default = 'no'
        )
    parser.add_argument(
        '--temp',
        dest = 'temperature',
        type = float,
        default = .5
        )
    parser.add_argument(
        '--max_new_tokens',
        dest = 'max_new_tokens',
        type = int,
        default = 256
        )
    parser.add_argument(
        '--batch_size',
        dest = 'batch_size',
        type = int,
        default = 10
        )
    parser.add_argument(
        '--start_from_batch',
        dest = 'start_from_batch',
        type = int,
        default = 0
        )
    args = parser.parse_args()
    
    querier = ModelQuerier.ModelQuerier(
                                        model_name = args.model, 
                                        language = args.language,
                                        allocate = args.allocate,
                                        max_new_tokens = args.max_new_tokens, 
                                        temperature = args.temperature
                                    )
    
    model_name = args.model if querier.is_gpt else args.model.split('/')[-1]
    output_path = os.path.join(OUTPUT_ROOT, args.language, args.judgment_type)
    os.makedirs(output_path, exist_ok = True)

    #     "deepseek-ai/deepseek-coder-1.3b-instruct",
    #     "deepseek-ai/deepseek-coder-6.7b-instruct", 
    #     "deepseek-ai/deepseek-coder-33b-instruct",
    #     "codellama/CodeLlama-7b-Instruct-hf",
    #     "codellama/CodeLlama-13b-Instruct-hf",
    #     "codellama/CodeLlama-34b-Instruct-hf",
    #     "gpt-3.5-turbo",
    #     "gpt-4-turbo"


    #import data
    data = pd.read_csv(os.path.join(INPUT_PATH, f'CS-benchmark-{args.language.capitalize()}.csv'))

    batches = list(divide_into_batches(data, batch_size = args.batch_size))
    print(f'Evaluating {querier.model_name} as judge on code summarization.\n')
    
    overall_df = pd.DataFrame()
    for batch in tqdm(batches[args.start_from_batch:], total = len(batches), initial = args.start_from_batch, desc = "Batch", position = 0):
        batch = batch[['target_id', 'target', 'generated_by', 'summary', 'summary_postprocessed']]

        prompts = []
        model_outputs = []
        for instance in tqdm(batch.itertuples(index = False), desc = "Instance", leave = False):
            method = instance.target
            summary = instance.summary_postprocessed
            
            prompt, model_output = querier.judge_code_summary(
                                                                method = method,
                                                                summary = summary,
                                                                judgment_type = args.judgment_type
                                                            )
            prompts.append(prompt)
            model_outputs.append(model_output)

        batch['prompt'] = prompts
        batch['model_output'] = model_outputs

        # saving results to file
        overall_df = pd.concat([overall_df, batch])
        if args.start_from_batch == 0:
            overall_df.to_csv(os.path.join(output_path, f'{model_name}.csv'), index = False)
        else:
            overall_df.to_csv(os.path.join(output_path, f'{model_name}_{now}.csv'), index = False)

    print()