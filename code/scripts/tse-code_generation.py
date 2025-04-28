import json
import os
import subprocess
import argparse
import ModelQuerier
import ResultExtractor
import sys
import time


INPUT_PATH = f'../data/input'
LANGUAGE = 'java'
OUTPUT_PATH = f'../data/predictions/{LANGUAGE}'
CODEREVAL_IDS_TODISCARD = json.load(open('../constants/ids_to_discard.json'))[f"CoderEval {LANGUAGE.capitalize()} Ids with Unreliable Tests"]

def sleep(sleeping_time_in_seconds):
    for remaining in range(sleeping_time_in_seconds, 0, -1):
        sys.stdout.write("\r")
        sys.stdout.write("{:2d} seconds remaining...".format(remaining))
        sys.stdout.flush()
        time.sleep(1)

    sys.stdout.write("\r\n")
    

if __name__ == '__main__':

    os.makedirs(OUTPUT_PATH, exist_ok = True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action = "store", dest = 'model')
    parser.add_argument('--temp', action = "store", dest = 'temperature', type = float)
    parser.add_argument('--beam', action = "store", dest = 'beam', default = 1, type = int)
    parser.add_argument('--allocate', action = "store", dest = 'allocate', default = False, type = bool)
    parser.add_argument('--sleep_time_s', action = "store", dest = 'sleep', default = 0, type = int)
    args = parser.parse_args()


    #import methods
    instances = json.load(open(os.path.join(INPUT_PATH, f'CoderEval4{LANGUAGE.capitalize}.json')))['RECORDS']

    instances = [inst for inst in instances if inst['_id'] not in CODEREVAL_IDS_TODISCARD]
    
    # model = "deepseek-ai/deepseek-coder-1.3b-instruct"
    # model = "deepseek-ai/deepseek-coder-6.7b-instruct"
    # model = "deepseek-ai/deepseek-coder-33b-instruct"
    # model = "codellama/CodeLlama-7b-Instruct-hf"
    # model = "codellama/CodeLlama-13b-Instruct-hf"
    # model = "codellama/CodeLlama-34b-Instruct-hf"
    # model = "gpt-turbo-4"
    
    querier = ModelQuerier.ModelQuerier(model = args.model, allocate = args.allocate, temperature = args.temperature)
    
    code_extractor = ResultExtractor.ResultExtractor()

    file_extention = 'py' if LANGUAGE == 'python' else LANGUAGE
    
    
    print(f'Testing {args.model} on {LANGUAGE.upper()}.')

    result_dict = {}
    for ibeam in range(args.beam):
        for iinstance in range(len(instances)):
            print(f'\rBeam {ibeam + 1}/{args.beam}, Instance: {iinstance + 1}/{len(instances)}', end = '')

            if querier.is_gpt:
                model_output = querier.codegeneration_codereval_chatgpt(instances[iinstance], language = LANGUAGE)
                predicted_method = code_extractor.extract_predicted_method_from_output(model_output, file_extention = file_extention, gpt_flag = querier.is_gpt)
            
            elif not args.allocate:
                model_output = querier.codegeneration_codereval(instances[iinstance], language = LANGUAGE)
                signature = ModelQuerier.extract_signature_codereval(instances[iinstance], LANGUAGE)
                predicted_method = code_extractor.extract_predicted_method_from_output(model_output, 
                                                                                       file_extention, 
                                                                                       gpt_flag = querier.is_gpt, 
                                                                                       hat = signature,
                                                                                       tempfile_name = instances[iinstance]['_id']
                                                                                    )
            elif args.allocate:
                model_output = querier.codegeneration_codereval(instances[iinstance], language = LANGUAGE)
                signature = ModelQuerier.extract_signature_codereval(instances[iinstance], LANGUAGE)
                predicted_method = code_extractor.extract_predicted_method_from_output(model_output, 
                                                                                       file_extention, 
                                                                                       gpt_flag = querier.is_gpt, 
                                                                                    #    hat = signature,
                                                                                       tempfile_name = instances[iinstance]['_id']
                                                                                    )

            method_id = instances[iinstance]['_id']            
            result_dict.setdefault(iinstance, {})
            result_dict[iinstance].setdefault('_id', method_id)
            result_dict[iinstance].setdefault('generate_results', [])
            result_dict[iinstance]['generate_results'].append(predicted_method)
        
        sleep(args.sleep)
    
    model_name = args.model.split('/')[1] if not querier.is_gpt else args.model
    
    os.makedirs(OUTPUT_PATH, exist_ok = True)
    with open(os.path.join(OUTPUT_PATH, f'{model_name}.jsonl'), 'w') as fout:
        for iinstance, dict_to_write in result_dict.items():
            json.dump(dict_to_write, fout)
            fout.write('\n') if iinstance != len(result_dict) - 1 else fout.write('')
    fout.close()
    
    print()
    subprocess.call(f'rm ../data/temp/*', shell = True)