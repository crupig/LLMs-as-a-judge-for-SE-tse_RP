import os
import pandas as pd
import re

INPUT_PATH = '../data/results/cs'

if __name__ == '__main__':

    pattern_dict = {
        
        'CodeLlama-7b-Instruct-hf' : [
            r"\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\s*([1-5])",
            r"\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\n\n\* Rating:\s*([1-5])",
            r"[1-3]\.\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\n\nRating:\s*([1-5])",
            r"[1-3]\.\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\s*([1-5])",
            
            r"the (Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability) as ([1-5])",
            r"The Comment is rated ([1-5]) out of 5 for (Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability)"
            
        ],
        
        'CodeLlama-13b-Instruct-hf' : [
            r"#{2}\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\n\n\*? ?Rating:\s*([1-5])",
            r"(?!\*|[1-3]\.)\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\s*([1-5])",
            r"[1-3]\.\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\nRating:?\s*([1-5])",
            
            r"the (Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability) as ([1-5])"
        ],
        
        'CodeLlama-34b-Instruct-hf' : [
            r"#{2}\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\n\n\*? ?Rating:\s*([1-5])",
            r"(?!\*|[1-3]\.)\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\s*([1-5])",
        ],

        'gpt-3.5-turbo' : [
            r"\#\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\n#{2}\s*\#?\s*Rating:\s*([1-5])",
            
            r"\#\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\n([1-5])",
            r"#{2}\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\n#{3}\s*\#?\s*Rating:\s*([1-5])",
        ],

        'gpt-4-turbo' : [
            r"(?!#{2}|#{3})\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\n\*\*\s*\#?\s*Rating:\s*([1-5])",
            r"#{3}\s*[1-3]\.\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\n\*\*\s*\#?\s*Rating:\s*([1-5])",
            r"#{3}\s*[1-3]\.\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\n#{4}\s*Rating:\s*([1-5])",
            r"#{3}\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\n#{4}\s*Rating:?\n([1-5])",
            r"#{2}\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\n#{3}\s*Rating:?\n([1-5])",
            r"#{2}\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\n#{3}\s*Rating:?\s*([1-5])",
            r"#{3}\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\n#{4}\s*Rating:?\s*([1-5])",

            r"\#\s*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\n\*\*\s*Rating:?\*\*:?\s*([1-5])",
            r"\*\*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability) Rating:?\s*([1-5])",
            r"\*\*(Content Adequacy|Conciseness|Fluency & Understandability|Fluency and Understandability):?\*\*\nRating:?\s*([1-5])",
        ]
    }


    for file in sorted([f for f in os.listdir(INPUT_PATH) if os.path.isfile(os.path.join(INPUT_PATH, f))]):
        
        model_name = file.split('_')[0]

        print(f'Writing: {file}')

        judge_output_df = pd.read_csv(os.path.join(INPUT_PATH, file))
        judge_output_df.fillna('', inplace = True)
        
        content_adequacy = []
        conciseness = []
        fluency = []

        for i in range(judge_output_df.shape[0]):
            ca_hasmatched, con_hasmatched, flu_hasmatched = False, False, False
            judgement = judge_output_df['model_output'].iloc[i]
            judgement = judgement.replace('/5', '')
                
            for patt in pattern_dict[model_name]:
                pattern = re.compile(patt, re.IGNORECASE)
                matches = pattern.findall(judgement)
                
                for match in matches:
                    aspect = match[0].lower()
                    aspect = aspect.replace('&', 'and')
                    if aspect == 'content adequacy' and not ca_hasmatched:
                        content_adequacy.append(match[1].strip())
                        ca_hasmatched = True
                    if aspect == 'conciseness' and not con_hasmatched:
                        conciseness.append(match[1].strip())
                        con_hasmatched = True
                    if aspect == 'fluency and understandability' and not flu_hasmatched:
                        fluency.append(match[1].strip())
                        flu_hasmatched = True

                if  ca_hasmatched == True and con_hasmatched == True and flu_hasmatched == True:
                    break

            if ca_hasmatched == False:
                content_adequacy.append('-')
            if con_hasmatched == False:
                conciseness.append('-')
            if flu_hasmatched == False:
                fluency.append('-')
            
        d = {
            f'{model_name}_CA' : content_adequacy,
            f'{model_name}_Conciseness' : conciseness,
            f'{model_name}_Fluency' : fluency,
        }
        
        to_concat = pd.DataFrame(d)
        out = pd.concat([judge_output_df, to_concat], axis = 1)
        out.to_csv(os.path.join(INPUT_PATH, f'{file}'), index = False)