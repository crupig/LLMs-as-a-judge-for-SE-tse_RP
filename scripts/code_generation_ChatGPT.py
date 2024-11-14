from tqdm import tqdm
import os
import json
import protocols

INPUT_PATH = '../data/input'
INPUT_FILE = 'CoderEval4Java.json'

if __name__ == '__main__':

    #import java methods
    instances = json.load(open(os.path.join(INPUT_PATH, INPUT_FILE)))['RECORDS']
    
    # gpt_version = 'gpt-3.5-turbo'
    gpt_version = 'gpt-4-turbo'

    output_jsonl = []
    language = INPUT_FILE.split('4')[1].split('.')[0].lower()
    file_extention = 'py' if language == 'python' else language
    print(f'Testing ChatGPT, version: {gpt_version} on {language.upper()}.')
    
    for i in tqdm(range(len(instances))):
        docstring = instances[i]['human_label']
        first_line = instances[i]['code'].split('\n')[0].strip()

        if file_extention == 'py' and len(first_line) > 0 and first_line[0] == '@':
            signature = '\n'.join([line.strip() for line in instances[i]['code'].split('\n')[:2]])
        elif file_extention == 'py' and len(first_line) > 0 and first_line.split(' ')[0] == 'import':
            signature = instances[i]['code'].split('\n')[1].strip()
        else:
            signature = instances[i]['code'].split('\n')[0].strip()
        
        method_id = instances[i]['_id']
        
        prompt = f"""Implement the following {language} method.
Description: "{docstring}"
Signature: "{signature}"
Only output the method implementation including the signature, and no other text."""
        
        gpt_output = protocols.ask_chatgpt(prompt, gpt_version, language)
        predicted_method = protocols.extract_predicted_method_from_output(model = gpt_version, model_output = gpt_output, file_extention = file_extention, gpt_flag = True)
        predicted_method = predicted_method if predicted_method != [] else ''

        to_jsonl = {}
        to_jsonl.setdefault('_id', method_id)
        to_jsonl.setdefault('generate_results', [])
        to_jsonl['generate_results'].append(predicted_method)
        output_jsonl.append(to_jsonl)

    with open(f'../data/results/code_generation/{language}/{gpt_version}.jsonl', 'w') as fout:
        for i, j in enumerate(output_jsonl):
            json.dump(j, fout)
            fout.write('\n') if i != len(output_jsonl) - 1 else fout.write('')
    fout.close()