import os
import pandas as pd
import re

INPUT_PATH = '../data/results/cg_judgement/java/scale'

if __name__ == '__main__':

    for file in sorted([f for f in os.listdir(INPUT_PATH) if '.csv' in f and 'batch' not in f and '_RR' not in f and 'Repair' not in f]):
        
        df = pd.read_csv(os.path.join(INPUT_PATH, file))
        df = df.fillna('')
        
        method_id = []
        judged_model = []
        rating = []
        rationale = []

        for model_col in [c for c in df.columns if '_judgement' in c]:
        
            for i in range(df.shape[0]):
                patterns = [
                    r"# Rating\s*(?P<rating>\d+)\s*# Rationale\s*(?P<rationale>.+)",
                    r"# Rationale\s*(?P<rationale>.+)\s*# Rating\s*(?P<rating>\d+)",
                    r"(?:#\s*)?Rating:\s*(?P<rating>\d+)\s*(?:#\s*)?Rationale:\s*(?P<rationale>.+)",
                    r"(?:#\s*)?Rationale:\s*(?P<rationale>.+)\s*(?:#\s*)?Rating:\s*(?P<rating>\d+)",
                    r"# Rating\s*(?P<rating>1\.\*\*Very unlikely to be correct\*\*|2\.\*\*Unlikely to be correct\*\*|3\.\*\*Undecided\*\*|4\.\*\*Likely to be correct\*\*|5\.\*\*Very likely to be correct\*\*)\s*# Rationale\s*(?P<rationale>.+)",
                    r"# Rationale\s*(?P<rationale>.+)\s*# Rating\s*(?P<rating>1\.\*\*Very unlikely to be correct\*\*|2\.\*\*Unlikely to be correct\*\*|3\.\*\*Undecided\*\*|4\.\*\*Likely to be correct\*\*|5\.\*\*Very likely to be correct\*\*)",
                ]
                
                for patt_idx, pattern in enumerate(patterns):
                    pattern = re.compile('{}'.format(pattern))
                    
                    judgement = df[model_col].iloc[i]
                    judgement = judgement.replace('/5', '')

                    match = pattern.search(judgement)

                    if match:
                        rate = match.group("rating")[0] if len(match.group("rating")) > 1 else match.group("rating")
                        ratio = match.group("rationale").strip()
                        rating.append(rate)
                        rationale.append(ratio)
                        
                        method_id.append(df["id"].iloc[i])
                        judged_model.append(model_col.split("_judgement")[0])
                        break

                    elif patt_idx != len(patterns) - 1:
                        continue
                    
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