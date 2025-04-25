import json
import os
import protocols

INPUT_PATH = '../data/input'
INPUT_FILE = 'CoderEval4Java.json'

if __name__ == '__main__':

    #import java methods
    instances = json.load(open(os.path.join(INPUT_PATH, INPUT_FILE)))['RECORDS']
    
    model = "deepseek-ai/deepseek-coder-1.3b-instruct"
    # model = "deepseek-ai/deepseek-coder-6.7b-instruct"
    # model = "deepseek-ai/deepseek-coder-33b-instruct"
    # model = "codellama/CodeLlama-7b-Instruct-hf"
    # model = "codellama/CodeLlama-13b-Instruct-hf"
    # model = "codellama/CodeLlama-34b-Instruct-hf"

    output_jsonl = []
    language = INPUT_FILE.split('4')[1].split('.')[0].lower()
    file_extention = 'py' if language == 'python' else language
    print(f'Testing {model} on {language.upper()}.')

    for j in range(len(instances)):
        print(f'\rInstance: {j + 1}/{len(instances)}', end = '')
        
        method_id = instances[j]['_id']
        predicted_method = protocols.ask4prediction(model, instances[j], file_extention = file_extention)
        
        to_jsonl = {}
        to_jsonl.setdefault('_id', method_id)
        to_jsonl.setdefault('generate_results', [])
        to_jsonl['generate_results'].append(predicted_method)
        output_jsonl.append(to_jsonl)

    model_name = model.split('/')[1]
    with open(f'../data/results/code_generation/{language}/{model_name}.jsonl', 'w') as fout:
        for i, j in enumerate(output_jsonl):
            json.dump(j, fout)
            fout.write('\n') if i != len(output_jsonl) - 1 else fout.write('')
    fout.close()
    
    print()