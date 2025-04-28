import json
import os
import pandas as pd
import argparse
import ModelQuerier
import ResultExtractor
from tqdm import tqdm
from datetime import datetime

DEBUG_MODE = '../'

INPUT_PATH = f'{DEBUG_MODE}data/input/cg'

def divide_into_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]

def extract_signature_py(code):
    for line in code.split('\n'):
        if line.strip().startswith('def'):
            return line.strip()

def same_signature(row):
    sig1 = extract_signature_py(row['generated_code'])
    sig2 = extract_signature_py(row['target'])
    return sig1 == sig2

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
        choices = ['bool', 'scale', 'slowthinking', 'stepbystep'],
        dest = 'judgment_type',
        help = "Choose one of the following values: 'bool', 'scale', 'slowthinking', 'stepbystep'."
        )
    parser.add_argument(
        '--rationale',
        choices = ['yes', 'no'],
        dest = 'rationale',
        help = "Choose one of the following values: 'yes', 'no'.",
        default = 'yes'
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

    OUTPUT_ROOT = f'{DEBUG_MODE}data/results/tse/tse_maj/{args.language}'
    INCOMPLETE_IMPLEMENTATIONS = pd.read_csv(f'{DEBUG_MODE}data/input/{args.language}_incomplete.csv')
    INCOMPLETE_IMPLEMENTATIONS['id_generatedby'] = INCOMPLETE_IMPLEMENTATIONS['target_id'] + '_' + INCOMPLETE_IMPLEMENTATIONS['generated_by']
    INCOMPLETE_IMPLEMENTATIONS = INCOMPLETE_IMPLEMENTATIONS['id_generatedby'].tolist()

    os.chdir('/home/giuseppe/benchmarks')
    IDS_TO_DISCARD = json.load(open(f'ids_to_discard.json'))[f"CoderEval {args.language.capitalize()} Ids with Unreliable Tests"]
    os.chdir('/home/giuseppe/llms_as_judge/scripts')

    #import data
    data = json.load(open(os.path.join(INPUT_PATH, f'CoderEval4{args.language.capitalize()}.json')))['RECORDS']
    data = [inst for inst in data if inst['_id'] not in IDS_TO_DISCARD]
    
    # judge_model = "deepseek-ai/deepseek-coder-1.3b-instruct"
    # judge_model = "deepseek-ai/deepseek-coder-6.7b-instruct"
    # judge_model = "deepseek-ai/deepseek-coder-33b-instruct"

    # judge_model = "codellama/CodeLlama-7b-Instruct-hf"
    # judge_model = "codellama/CodeLlama-13b-Instruct-hf"
    # judge_model = "codellama/CodeLlama-34b-Instruct-hf"

    # judge_model = "gpt-3.5-turbo"
    # judge_model = "gpt-4-turbo"

    querier = ModelQuerier.ModelQuerier(
                                        model_name = args.model, 
                                        language = args.language,
                                        allocate = args.allocate,
                                        max_new_tokens = args.max_new_tokens, 
                                        temperature = args.temperature
                                    )
    extractor = ResultExtractor.ResultExtractor()
    
    generator_models = [
        "deepseek-coder-1.3b-instruct",
        "deepseek-coder-6.7b-instruct", 
        "deepseek-coder-33b-instruct",
        "CodeLlama-7b-Instruct-hf",
        "CodeLlama-13b-Instruct-hf",
        "CodeLlama-34b-Instruct-hf",
        "gpt-3.5-turbo",
        "gpt-4-turbo"
    ]

    prediction_path = f'{DEBUG_MODE}data/results/icse25/code_generation/{args.language}'
    
    batches = list(divide_into_batches(data, batch_size = args.batch_size))
    print(f'Testing {args.model} as judge on {args.language.upper()}.')

    overall_df = pd.DataFrame()
    for batch in tqdm(batches[args.start_from_batch:], total = len(batches), initial = args.start_from_batch, desc = "Batch", position = 0):
        batch_ids = [i['_id'] for i in batch]
        batch_target = [i['code'] for i in batch]
        batch_df = pd.DataFrame({'id' : batch_ids, 'target' : batch_target})

        # generate judgments for automatically generated methods
        for generator in tqdm(generator_models, desc = "Generator", leave = False): # loop over generators models
            predictions = extractor.extract_predictions_for_codereval(os.path.join(prediction_path, f'{generator}.jsonl'), batch_ids)
            predictions = predictions.merge(batch_df.rename(columns = {'id' : 'target_id'}), on = 'target_id', how = 'left')
            predictions['id_generatedby'] = predictions['target_id'] + '_' + generator
            if args.language == 'python':
                predictions['same_sign'] = predictions.apply(same_signature, axis = 1)
            else:
                predictions['same_sign'] = True
            
            model_code_dict = {}
            model_output_dict = {}
            model_prompt_dict = {}

            for instance in tqdm(predictions.itertuples(index = False), desc = "Instance", leave = False): # loop over predictions of a single generator model
                if instance.generated_code == '' or instance.same_sign == False or instance.id_generatedby in INCOMPLETE_IMPLEMENTATIONS:
                    model_code_dict.setdefault(generator, []).append("<PLACEHOLDER>")
                    model_output_dict.setdefault(generator, []).append("<PLACEHOLDER>")
                    model_prompt_dict.setdefault(generator, []).append("<PLACEHOLDER>")
                else:
                    target_instance = [inst for inst in batch if inst['_id'] == instance.target_id][0]
                    model_code_dict, model_output_dict, model_prompt_dict = querier.judge_code_correctness_codereval(
                            target_instance = target_instance,
                            generator = generator,
                            candidate = instance.generated_code,
                            judgment_type = args.judgment_type,
                            rationale = args.rationale,
                            model_code_dict = model_code_dict, 
                            model_output_dict = model_output_dict, 
                            model_prompt_dict = model_prompt_dict
                        )
            
            to_concat = pd.DataFrame({
                f'{generator}_prompt' : list(model_prompt_dict.values())[0],
                f'{generator}_prediction' : list(model_code_dict.values())[0],
                f'{generator}_judgement' : list(model_output_dict.values())[0]
            })
            batch_df = pd.concat([batch_df, to_concat], axis = 1)

        
        # generate judgments for human written methods
        model_code_dict = {}
        model_output_dict = {}
        model_prompt_dict = {}
        for target_instance in batch:
            model_code_dict, model_output_dict, model_prompt_dict = querier.judge_code_correctness_codereval(
                    target_instance = target_instance,
                    generator = 'humanwritten',
                    candidate = target_instance['code'],
                    judgment_type = args.judgment_type,
                    rationale = args.rationale,
                    model_code_dict = model_code_dict, 
                    model_output_dict = model_output_dict, 
                    model_prompt_dict = model_prompt_dict
                )
        
        to_concat = pd.DataFrame({
            f'humanwritten_prompt' : list(model_prompt_dict.values())[0],
            f'humanwritten_prediction' : list(model_code_dict.values())[0],
            f'humanwritten_judgement' : list(model_output_dict.values())[0]
        })
        batch_df = pd.concat([batch_df, to_concat], axis = 1)
        

        # saving results to file
        model_name = args.model if 'gpt-' in args.model else args.model.split('/')[-1]
        output_folder = args.judgment_type
        output_path = os.path.join(OUTPUT_ROOT, output_folder)
        os.makedirs(output_path, exist_ok = True)
        overall_df = pd.concat([overall_df, batch_df])
        if args.start_from_batch == 0:
            overall_df.to_csv(os.path.join(output_path, f'{model_name}.csv'), index = False)
        else:
            overall_df.to_csv(os.path.join(output_path, f'{model_name}_{now}.csv'), index = False)

    print()