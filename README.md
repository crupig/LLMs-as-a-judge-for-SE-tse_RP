# LLMs-as-a-judge-for-SE
This repository is the replication package of the work **"On the Effectiveness of LLM-as-judge for Software-related Tasks"**. The purpose of this repository is to provide the data and discuss the pipeline that we used to run this study.

## LLM as a judge on Code Generation task
### Pipeline

For this part of our work we rely on the **CoderEval ICSE'24** dataset for **Java**, which can be found [here](https://github.com/CoderEval/CoderEval).

**1) Dataset cleaning:**

Clean the CoderEval dataset from instances with noisy or wrong test cases:
- remove instances which do not pass the test;
- remove the instances which pass the test with an empty body (only one comment is present);
- remove instances which pass the test with a trivial body (only the return statement is present);
  
Let ```N``` be the number of valid instances present in the CoderEval dataset.

**2) Code generation (```code_generation.py```/```code_generation_ChatGPT.py```):**
- for each of the ```N``` valid instances, we get the predictions from 8 LLMs (```deepseek-ai/deepseek-coder-1.3b-instruct```, ```deepseek-ai/deepseek-coder-6.7b-instruct```, ```deepseek-ai/deepseek-coder-33b-instruct```, ```codellama/CodeLlama-7b-Instruct-hf```, ```codellama/CodeLlama-13b-Instruct-hf```, ```codellama/CodeLlama-34b-Instruct-hf```, ```gpt-3.5-turbo```, ```gpt-4-turbo```);
- the prompt for the code generation task (for all the LLMs not belonging to the ChatGPT family) is: ```{docstring}\n{signature}```;
- the prompt for the GPT models is : ```Implement the following Java method.\nDescription: "{docstring}"\nSignature: "{signature}"\nOnly output the method implementation including the signature, and no other text.```;

Since some LLMs may fail at the code generation task (i.e., from their output no valid methods can be extracted), the total amount of candidates is at most ```N x number_LLMs```. Let ```M``` be the total number of valid candidates.

**3) Outcome of CoderEval tests:**

The candidates go through the CoderEval test cases and for each of them we associate a 1 if the candidate passes the tests and 0 otherwise. Information about the outcome of the CoderEval tests can be found in the ```cg_judgement_java_boolean_human.csv``` file.
  	
**4) Each LLM acts as a judge on the code generation task (```judge_code_generation.py```):**

A chosen model as judge is asked to assert the quality of a given candidate based on the description and the signature of a target method. The judge is asked to provide a binary answer (i.e., 1 if it thinks that the candidate is correct and 0 otherwise). The judge is asked both for a ```#Rating``` and a ```#Rationale``` (i.e., an explanation of the ```#Rating```). Each model as judge will evaluate ```M``` candidates. The optimized prompt that we fed to the LLMs as a judge is ```prompts/prompt-judge-code-generation-boolean.tex```.

**5) Extract the ```#Rating``` and the ```#Rationale``` for each judgement (```extract_rating_rationale_bool.py```):**

This script is used to extract the ```#Rating``` and ```#Rationale``` information from the output of the models as judges. For each model some manually crafted heuristics are applied to the ```M``` model outputs.

**6) Rating and Rationale manual check:**

All outputs from point ```5)``` are manually analyzed to extract information that were missed by the heuristics. Note that, in the cases in which the judge fails to provide a valid judgement (i.e., either the ```#Rating``` or the ```#Rationale``` are not present in the model output), both the ```#Rating``` and the ```#Rationale``` are set to "-".

Results are in ```data/results/cg_judgement_java_boolean_human.csv```. Note that to the ```M``` judgements per LLM we added the judgements on the target methods (which pass the test of CoderEval by construction); Therefore, each LLM as judge was asked to judge ```M + N``` different candidates in total.

**7) Quantitative analysis:**

Our quantitative analysis can be found in the notebook ```notebooks/visualization_cg.ipynb```.

**8) Qualitative manual analysis of judgement failure cases:**

The goal is to extract a sample of cases in which each LLM as judge fails at judging a candidate method. For each judge we sample 15 false positives (i.e., cases in which the candidate is evaluated positively by the judge, but does not pass the relative test of the CoderEval dataset) and false negatives (i.e., candidates that are evaluated negatively by the judge, but do pass the relative test of the CoderEval dataset). We sample 15 examples (when present) per judge and per category of failure (false positives/negatives). We define a set of categories for which the judge may fail at judging the candidate method and then we assign each case of failure to one or more of our categories. The output of our manual analysis is reported in ```cg_MA.csv```, whereas the number of occurrences of each category is reported in ```cg_MA_false_positives.csv``` and ```cg_MA_false_negatives.csv```. These files are in the ```data/results``` folder.

## Creation of the Code Summarization benchmark for Java

For this part of our work we create a code summarization benchmark for Java starting from the CoderEval dataset discussed above. Our dataset features human judgements of 594 summaries. To build the dataset, we selected from the CoderEval benchmark the top-100 Java methods in terms of number of statements they feature. We decided to focus on the longest methods since those are the ones for which a good summary is likely to make a difference in terms of code comprehensibility and, thus, assessing the quality of summaries for these methods may make more sense. Among these 100 methods we found one that was a duplicate and was thus removed from the set, leaving us with 99 methods. For each of them, we have the associated code summary written by the original developer of the method. Also, we asked five LLMs (i.e., CodeLlama 7B, 13B, and 34B, GPT-3.5-turbo and GPT-4-turbo) to generate a summary for each of these 99 methods, leading to the total of 99 (manually written) + 99×5 automatically generated) = 594 summaries. The prompt used to generate code summaries with the LLMs is documented in ```prompts/prompt-summary-generation.tex```. The dataset is available at ```data/cs_benchmark/benchmark.csv```.

## LLM as judge on Code Summarization task
### Pipeline

We use the dataset described before to run our judgements on code summarization.

**1) LLMs as a judge for code summarization (```judge_code_summarization.py```):**

- for each snippet-summary pair, a chosen model as judge is asked to assert the quality of the summary with respect to the snippet. The judge is asked to give a ```#Rating``` and a ```#Rationale``` for 3 different aspects: ```content adequacy```, ```conciseness``` and ```fluency & understandability```. The optimized prompt that we fed to the LLMs as judge is reported in ```prompts/prompt-judge-code-summarization.tex```;
- for this task we select only 5 LLMs which will play the role of the judge (namely ```codellama/CodeLlama-7b-Instruct-hf```, ```codellama/CodeLlama-13b-Instruct-hf```, ```codellama/CodeLlama-34b-Instruct-hf```, ```gpt-3.5-turbo```, ```gpt-4-turbo```) because unfortunately the LLMs belonging to the DeepSeek Coder family often give an invalid output.
- the script to collect judgments for code summarization is ```judge_code_summarization.py```.

**2) ```#Rating``` extraction for all quality aspects (```extract_rating_ccf.py```):**

- a set of heuristics is used to extract the values for ```#Rating``` for all the quality aspects (```content adequacy```, ```conciseness``` and ```fluency & understandability```) from the output of the LLMs;
- the script used to do so is ```extract_rating_ccf.py```.
- manual extraction is performed when the heuristics fail;

**3) Quantitative analysis:**

Our quantitative analysis can be found in the notebook ```notebooks/visualization_cs.ipynb```.

## Data (```data/results```):

**1) ```cg_judgement_java_boolean_human.csv``` and ```cg_judgement_java_scale_human.csv```:**

These are the overall outputs of the code generation judgement task. They are spreadsheets containing the following fields:

- ```target_id``` : alphanumeric string associated to each valid instance of CoderEval;
- ```generated_by``` : the LLM which generated the candidate code (or ```human_written``` if the method was the target);
- ```generated_code``` : the candidate method (or target method);
- ```is_pass``` : 1 if the candidate passes the test of the CoderEval dataset, 0 otherwise (target methods are associated with 1 by construction);
- ```{LLM1}_rating``` : the rating given by the model as judge;
- ```{LLM1}_rationale``` : the rationale given by the model as judge;
- ...

The last two columns are repeated for each LLM as judge.
Note that two version of this file are present: boolean and 5-level (scale) rating. These files are reported in the in the ```data/results``` folder.

**2) ```cg_MA.csv```:**

Results of the manual analysis for code generation judgement failures.

**3) ```cg_MA_false_positives.csv``` and ```cg_MA_false_negatives.csv```:**

The most frequent reasons (i.e., categories) why LLMs as judges fail at the code generation judgment task are reported in these files.

**4) ```cs_{LLM_name}-raw-output.csv```:**

Contains the raw output of the models as judges (see point 2 of the Code Summarization pipeline for more information).

**5) ```cs_results-grouped.csv```:**

This is the overall output of the code summarization judgement task. It is a spreadsheet containing the following fields:

- ```question_id``` : id associate to each unique code snippet;
- ```mid``` : id associated to the technique which generated the summary of the code snippet (0 was assigned to human developers, other numbers to a given automatic code summarization techinique);
- ```user_id``` : number of human evaluators who rated the three aspects of the comment: ```content adequacy```, ```conciseness```, ```fluency```;
- ```Dev_CA``` : average content adequacy score assigned by the human evaluators to the summary;
- ```Dev_Conciseness``` : same as above for conciseness;
- ```Dev_Fluency``` : same as above for fluency;
- ```codeFunction``` : code snippet;
- ```codeComment``` : code summary;
- ```CA_{LLM1}_rating``` : the rating given by the model as judge to the content adequacy of the summary;
- ```CA_{LLM1}_rationale``` : the rationale given by the model as judge to the content adequacy of the summary;
- ```Conciseness_{LLM1}_rating``` : same as above for conciseness;
- ```Conciseness_{LLM1}_rationale``` : same as above for conciseness;
- ```Fluency_{LLM1}_rating``` : same as above for fluency;
- ```Fluency_{LLM1}_rationale``` : same as above for fluency;
- ...

The last six columns are repeated for each LLM as judge.

**6) ```cg_statistical_tests_bool.csv``` and ```cg_statistical_tests_scale.csv```:**

The results of the statistical tests run to obtain the right section of ```Table III``` in the paper, for both the boolean (actually reported in the paper) and the 5-level cases (omitted due to lack of space, but reported in ```imgs/RP.pdf```).

## Images (```imgs/RP.pdf```):

**1) ```Figure 1```:**

Same boxplot as in ```Figure 1``` in the paper, but considering only methods with no external dependencies.

**2) ```Figure 2```:**

Same boxplot as in ```Figure 2``` in the paper, but considering only methods with no external dependencies.

**3) ```Table I```:**

Same table as ```Table II``` in the paper, but considering only methods with no external dependencies.

**4) ```Table II```:**

Same table as ```Table III``` in the paper, but considering the 5-level code generation judgement scenario.

