import os
import pandas as pd
import re

INPUT_PATH = '../data/results/cg_judgement/java/boolean'

if __name__ == '__main__':

    for file in sorted([f for f in os.listdir(INPUT_PATH) if '.csv' in f and 'batch' not in f and '_RR' not in f and 'Repair' not in f]):
        
        df = pd.read_csv(os.path.join(INPUT_PATH, file))
        
        method_id = []
        judged_model = []
        rating = []
        rationale = []

        for model_col in [c for c in df.columns if '_judgement' in c]:
        
            for i in range(df.shape[0]):
                pattern = re.compile(r"# Rating\s*(?P<rating>\d+)\s*# Rationale\s*(?P<rationale>.+)")
                
                judgement = df[model_col].iloc[i]
                judgement = judgement.replace('0.0', '0')
                judgement = judgement.replace('1.0', '1')
                # Search for the pattern in the text
                match = pattern.search(judgement)

                # Extract and print the fields if a match is found
                if match:
                    rate = match.group("rating")
                    ratio = match.group("rationale").strip()
                    rating.append(rate)
                    rationale.append(ratio)
                elif file == 'deepseek-coder-1.3b-instruct.csv':
                    pattern = re.compile(r"\*\*(?P<bool>Wrong|Correct) Implementation\*\*")
                    matches = pattern.findall(judgement)
                    if len(matches) == 1:
                        is_correct = matches[0]
                        pattern2 = re.compile(r"# Rationale\s*(?P<rationale>.+)")
                        match2 = pattern2.search(judgement)
                        rate = '1' if is_correct == 'Correct' else '0'
                        ratio = match2.group("rationale").strip()
                        rating.append(rate)
                        rationale.append(ratio)
                    else:
                        rating.append("-")
                        rationale.append("-")

                elif file == 'CodeLlama-13b-Instruct-hf.csv':
                    pattern = re.compile(r"(?P<bool>Wrong|Correct) Implementation")
                    matches = pattern.findall(judgement)
                    if len(matches) == 1:
                        is_correct = matches[0]
                        pattern2 = re.compile(r"# Rationale\s*(?P<rationale>.+)")
                        match2 = pattern2.search(judgement)
                        rate = '1' if is_correct == 'Correct' else '0'
                        ratio = match2.group("rationale").strip()
                        rating.append(rate)
                        rationale.append(ratio)
                    else:
                        rating.append("-")
                        rationale.append("-")
                else:
                    rating.append("-")
                    rationale.append("-")
                
                method_id.append(df["id"].iloc[i])
                judged_model.append(model_col.split("_judgement")[0])
        
        d = {
            'id' : method_id,
            'judged_model' : judged_model,
            'rating' : rating,
            'rationale' : rationale,
        }
        
        file_name = '{}_RR.csv'.format(file.split('.csv')[0])
        pd.DataFrame(d).to_csv(os.path.join(INPUT_PATH, file_name), index = False)
        print(f'Writing: {file_name}')