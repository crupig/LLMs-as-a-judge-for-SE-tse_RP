from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch
import json
import requests
import openai
from huggingface_hub import InferenceClient
from colorama import Fore
from pygments.lexers import JavaLexer
import os
import re

def extract_signature_codereval(instance, language):
    """
        Description of the Function:
            extracts the signature of a given CoderEval instance.

        Parameters:
            instance (dict): instance of the CoderEval dataset. Each instance is a dictionary containing information about the java method.;
            language (str): either 'java' or 'python'.

        Returns:
            str: signature of the method.
    """
    assert language in ['java', 'python'], "language must be either 'java' or 'python'."

    code = instance['code'].strip()

    if language == 'python':
        signature = []
        for line in code.splitlines():
            if "def" not in line:
                signature.append(line)
            elif "def" in line:
                signature.append(line)
                return '\n'.join(signature)
    else:
        return code.split('\n')[0].strip()

def remove_comments_from_python_code(code):
    code = re.sub(r'""".*?"""', '', code, flags = re.DOTALL)
    code = re.sub(r'\'\'\'.*?\'\'\'', '', code, flags = re.DOTALL)
    code = '\n'.join([line for line in code.splitlines() if not line.strip().startswith('#')])
    return code

class ModelQuerier:

    def __init__(
            self,
            model_name,
            language,
            allocate = False,
            max_new_tokens = 512,
            temperature = .5,
            top_p = .95,
            do_sample = True
        ):
        """
            Class Description:
                instances of this class are objects to query LLMs either via the HF inference endpoints or API.
            
            Attributes:
                constants (dict): contains tokens and access keys;
                prompts (dict): contains prompts for different predefined tasks;
                model (str): name of the model to query;
                endpoint_url (str): the URL of the inference endpoint of the model;
                max_new_tokens (int): maximum number of tokens generated by the model (input tokens excluded);
                temperature (float): temperature to set when quering the model.
        """
        assert language in ['java', 'python'], "language must be either 'java' or 'python'."

        self.constants = json.load(open('../constants/constants_json.json'))
        self.prompts = json.load(open('../constants/prompts.json'))
        self.headers = {
            "Accept" : "application/json",
            "Authorization" : f"Bearer {self.constants['Hugging Face Inference Endpoints Token']}",
            "Content-Type" : "application/json" ,
            "x-use-cache" : "false"
            }
        self.allocate = allocate
        self.language = language
        self.model_name = model_name
        self.is_gpt = True if 'gpt-' in model_name else False
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        
        if self.is_gpt:
            self.allocate = False
            # set the openai API key
            os.environ['OPENAI_API_KEY'] = self.constants['OpenAI API Access Key']
            apikey = os.environ.get('OPENAI_API_KEY')
            if not apikey:
                raise ValueError('ERROR: OPENAI_API_KEY is not set! Check your .env file.')
        elif not self.allocate and not self.is_gpt:
            self.endpoint_url = json.load(open('../constants/IEPmodels.json'))[self.model_name]
            self.prompt_with_template = json.load(open('../constants/prompt_with_template.json'))[self.model_name]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code = True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code = True, torch_dtype = torch.bfloat16).cuda()
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            

    # a method to replace the tags like <LANGUAGE> in the prompt with the actual language
    @staticmethod
    def replace_tags(prompt, **kwargs):
        '''
        Possible tags to replace:
            <LANGUAGE>: language of the code;
            <DESCRIPTION>: description of the code;
            <SIGNATURE>: signature of the code;
            <CANDIDATE>: candidate code to evaluate;
            <FUNCTION>: function to summarize;
            <COMMENT>: comment to evaluate;
            <ANALYSIS>: analysis of the candidate code;
        '''
        for name, value in kwargs.items():
            if name not in ['language', 'description', 'signature', 'candidate', 'function', 'comment', 'analysis']:
                raise ValueError(f'Unknown tag: {name}')
            prompt = prompt.replace(f'<{name.upper()}>', value)
        return prompt

    def query_huggingface_inference_endpoint(self, prompt):
        """
            Description of the Function:
                queries the inference endpoint.

            Parameters:
                prompt (str): model input;
                temperature (float): temperature to set when quering the model.

            Returns:
                str: output of the model (input excluded).
        """
        messages = [
            {"role" : "user", "content" : prompt}
        ]
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token = self.constants['Hugging Face Tokenizer Token'])
        prompt_with_template = tokenizer.apply_chat_template(messages, tokenize = False) \
            if self.prompt_with_template else prompt
        
        payload = {
            "inputs": prompt_with_template,
            "parameters": {
                "max_new_tokens" : self.max_new_tokens,
                "return_full_text": False,
                "do_sample": self.do_sample,
                "top_p": self.top_p,
                "temperature": self.temperature,
                "seed": 123
            }
        }

        model_output = requests.post(url = self.endpoint_url, headers = self.headers, json = payload).json()[0]['generated_text']

        return model_output.strip()


    def query_allocated_model(self, prompt):
        if self.tokenizer is None or self.model is None:
            print('Either model or tokenizer is not allocated.')
            return ''
        
        code_generator = pipeline('text-generation', model = self.model, tokenizer = self.tokenizer)
        model_output = code_generator(
                                    prompt, 
                                    max_new_tokens = self.max_new_tokens,
                                    return_full_text = False,
                                    do_sample = self.do_sample,
                                    top_p = self.top_p,
                                    temperature = self.temperature
                                )[0]['generated_text']
        
        return model_output.strip()
    

    def query_chatgpt(self, prompt):
        """
            Description of the Function:
                queries ChatGPT API with a given prompt and returns ChatGPT response.

            Parameters:
                prompt (str): ChatGPT input;
                language (str): string indicating a programming language;

            Returns:
                str: ChatGPT output.
        """
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model = self.model_name,
            messages = [
                {'role' : 'system', 'content' : f'You are an expert {self.language.capitalize()} developer'},
                {'role' : 'user', 'content' : prompt}
            ],
            temperature = self.temperature,
            seed = 123
        )

        model_output = response.choices[0].message.content
        return model_output.strip()


    def call_huggingface_model(self, prompt):
        """
            Description of the Function:
                given a model and a prompt calls the model through the API or Inference Endpoint and returns the output.

            Parameters:
                prompt (str): model input;
                task (str): just a string which explicits the task ("PREDICTION", "RANKING", "REFINEMENT") in case of failure.

            Returns:
                str: output of the model.
        """
        
        if self.endpoint_url == "API":
            try:
                client = InferenceClient(model = self.model_name, token = self.constants['Hugging Face API Token'])
                predicted_output = client.text_generation(prompt = prompt, max_new_tokens = self.max_new_tokens)
            except:
                print(Fore.RED + f'\n{self.model_name}: API FAILED.\n' + Fore.BLACK)
                predicted_output = f'{self.model_name}: API FAILED.'
        
        else:
            try:
                predicted_output = self.query_huggingface_inference_endpoint(prompt = prompt)
            except:
                print(Fore.RED + f'\n{self.model_name}: IEP FAILED.\n' + Fore.BLACK)
                predicted_output = f'{self.model_name}: IEP FAILED.'
        
        return predicted_output.strip('\n\t ')


    def codegeneration_codereval(self, target_instance):
        """
            Description of the Function:
                queries the model for the code generation task on a given instance of the CoderEval dataset.

            Parameters:
                language (str): string indicating a programming language;
                target_instance (dict): instance of the CoderEval dataset. Each instance is a dictionary containing information about the java method.

            Returns:
                str: raw output of the model.
        """
        docstring = target_instance['human_label']
        signature = extract_signature_codereval(target_instance, self.language)
        
        if self.is_gpt:
            prompt = self.prompts["Code Generation ChatGPT"]
            
        else:
            prompt = self.prompts["Code Generation"]
            lexer = JavaLexer()
            target_num_tokens = len(list(lexer.get_tokens(target_instance['code'])))
            self.max_new_tokens = 2 * target_num_tokens
            
        prompt = self.replace_tags(
            prompt = prompt,
            language = self.language.capitalize(),
            signature = signature,
            description = docstring
            )
        
        model_output = self.query_model(prompt = prompt)
        return model_output
        

    def query_model(self, prompt):
        if self.allocate:
            return self.query_allocated_model(prompt = prompt)
        elif self.is_gpt:
            return self.query_chatgpt(prompt = prompt)
        else:
            return self.call_huggingface_model(prompt = prompt)

    
    def summarygeneration(self, method):
        """
            Description of the Function:
                queries the model to generate a summary for a given method passed as input.

            Parameters:
                method (str): method to summarize.
            
            Returns:
                str: output of the model.
        """
        
        prompt = self.prompts["Summary Generation"]
        prompt = self.replace_tags(
            prompt = prompt,
            language = self.language.capitalize(),
            function = method
        )
        
        model_output = self.query_model(prompt = prompt)

        return model_output
    
    def judge_code_correctness_codereval(self, target_instance, generator, candidate, judgment_type, rationale, model_code_dict, model_output_dict, model_prompt_dict):
        """
            Description of the Function:
                queries the model for the code correctness judgment task on a given instance of the CoderEval dataset.

            Parameters:
                target_instance (dict): instance of the CoderEval dataset. Each instance is a dictionary containing information about the method;
                generator (str): automatic technique which generated the candidate to evaluate;
                candidate (str): the candidate to evaluate;
                judgment_type (str): either 'bool' or '5-level';
                rationale (str): either 'yes' or 'no';
                model_code_dict (dict): dictionary containing all the previous candidates judged by the model;
                model_output_dict (dict): dictionary containing all the previous outputs of the model;
                model_prompt_dict (dict): dictionary containing all the previous prompts of the model.

            Returns:
                dict: updated model_code_dict;
                dict: updated model_output_dict;
                dict: updated model_prompt_dict.
        """
        description = target_instance['human_label']
        signature = extract_signature_codereval(target_instance, self.language)
        
        if self.language == 'python':
            candidate = remove_comments_from_python_code(candidate)

        if judgment_type == 'slowthinking':
            model_code_dict, model_output_dict, model_prompt_dict = self.judge_code_correctness_slowthinking(
                description = description,
                signature = signature,
                candidate = candidate,
                generator = generator,
                model_code_dict = model_code_dict,
                model_output_dict = model_output_dict,
                model_prompt_dict = model_prompt_dict
            )

        elif judgment_type == 'stepbystep':
            model_code_dict, model_output_dict, model_prompt_dict = self.judge_code_correctness_stepbystep(
                description = description,
                signature = signature,
                candidate = candidate,
                generator = generator,
                model_code_dict = model_code_dict,
                model_output_dict = model_output_dict,
                model_prompt_dict = model_prompt_dict
            )

        else:
            model_code_dict, model_output_dict, model_prompt_dict = self.judge_code_correctness_zeroshot(
                judgment_type = judgment_type,
                rationale = rationale,
                description = description,
                signature = signature,
                candidate = candidate,
                generator = generator,
                model_code_dict = model_code_dict,
                model_output_dict = model_output_dict,
                model_prompt_dict = model_prompt_dict
            )

        return model_code_dict, model_output_dict, model_prompt_dict
    


    def judge_code_correctness_zeroshot(self, judgment_type, rationale, description, signature, candidate, generator, model_code_dict, model_output_dict, model_prompt_dict):
        judgment_type = 'Boolean' if judgment_type == 'bool' else '5-Level'
        rationale = ' No Rationale' if rationale == 'no' else ''
        prompt = self.prompts[f"Judge Code Generation {judgment_type}{rationale}"]
        prompt = self.replace_tags(
            prompt = prompt,
            language = self.language.capitalize(),
            description = description,
            signature = signature,
            candidate = candidate
        )

        model_output = self.query_model(prompt = prompt)
        
        model_code_dict.setdefault(generator, []).append(candidate)
        model_output_dict.setdefault(generator, []).append(model_output)
        model_prompt_dict.setdefault(generator, []).append(prompt)

        return model_code_dict, model_output_dict, model_prompt_dict


    def judge_code_correctness_slowthinking(self, description, signature, candidate, generator, model_code_dict, model_output_dict, model_prompt_dict):
        prompt_pt1 = self.prompts["Judge Code Generation Slow-Thinking Pt1"]
        prompt_pt1 = self.replace_tags(
            prompt = prompt_pt1,
            language = self.language.capitalize(),
            description = description,
            signature = signature,
            candidate = candidate
        )
        
        # first step of the slow-thinking process: ask for an analysis about the correctness of a candidate implementation
        analysis = self.query_model(prompt = prompt_pt1)

        if 'End of Evaluation' in analysis:
            analysis = analysis.split('End of Evaluation')[0].strip('# ')

        prompt_pt2 = self.prompts["Judge Code Generation Slow-Thinking Pt2"]
        prompt_pt2 = self.replace_tags(
            prompt = prompt_pt2,
            analysis = analysis
        )

        # second step of the slow-thinking process: take as input the analysis of the previous step and output "yes" if the analysis describes a correct candidate, and "no" otherwise
        max_new_tokens = self.max_new_tokens # stores the current value of max_new_tokens
        self.max_new_tokens = 32 # for the second step of the slow-thinking process, we set max_new_tokens to 128 (the model simply has to reply "yes" or "no")
        model_output = self.query_model(prompt = prompt_pt2)
        self.max_new_tokens = max_new_tokens # restore the original value of max_new_tokens

        model_code_dict.setdefault(generator, []).append(candidate)
        model_output_dict.setdefault(generator, []).append(f'{model_output}\n\n*************\n\n{analysis}')
        model_prompt_dict.setdefault(generator, []).append(f'{prompt_pt2}\n\n*************\n\n{prompt_pt1}')

        return model_code_dict, model_output_dict, model_prompt_dict
    
    
    def judge_code_correctness_stepbystep(self, description, signature, candidate, generator, model_code_dict, model_output_dict, model_prompt_dict):
        prompt_pt1 = self.prompts["Judge Code Generation Step-By-Step Pt1"]
        prompt_pt1 = self.replace_tags(
            prompt = prompt_pt1,
            language = self.language.capitalize(),
            description = description,
            signature = signature,
            candidate = candidate
        )

        reasoning = self.query_model(prompt = prompt_pt1)

        prompt_pt2 = self.prompts["Judge Code Generation Step-By-Step Pt2"]
        prompt_pt2 = self.replace_tags(
            prompt = prompt_pt2,
            analysis = reasoning
        )
        prompt_pt2 = prompt_pt1 + prompt_pt2
        
        max_new_tokens = self.max_new_tokens # stores the current value of max_new_tokens
        self.max_new_tokens = 128 # for the second step of the slow-thinking process, we set max_new_tokens to 128 (the model simply has to reply "yes" or "no")
        model_output = self.query_model(prompt = prompt_pt2)
        self.max_new_tokens = max_new_tokens # restore the original value of max_new_tokens

        model_code_dict.setdefault(generator, []).append(candidate)
        model_output_dict.setdefault(generator, []).append(f'{model_output}\n\n*************\n\n{reasoning}')
        model_prompt_dict.setdefault(generator, []).append(f'{prompt_pt2}\n\n*************\n\n{prompt_pt1}')

        return model_code_dict, model_output_dict, model_prompt_dict
    
    
    def judge_summary_quality_stepbystep(self, method, summary, judgment_type):
        prompt_pt1 = self.prompts["Judge Code Summarization Step-By-Step Pt1"]
        prompt_pt1 = self.replace_tags(
            prompt = prompt_pt1,
            language = self.language.capitalize(),
            function = method,
            comment = summary
        )

        reasoning = self.query_model(prompt = prompt_pt1)

        prompt_pt2 = self.prompts["Judge Code Summarization With Instructions Step-By-Step Pt2"] if 'extended_instructions' in judgment_type\
            else self.prompts["Judge Code Summarization Step-By-Step Pt2"]
        prompt_pt2 = self.replace_tags(
            prompt = prompt_pt2,
            analysis = reasoning
        )
        prompt_pt2 = prompt_pt1 + prompt_pt2
        
        max_new_tokens = self.max_new_tokens # stores the current value of max_new_tokens
        self.max_new_tokens = 128 # for the second step of the slow-thinking process, we set max_new_tokens to 128 (the model simply has to reply "yes" or "no")
        model_output = self.query_model(prompt = prompt_pt2)
        self.max_new_tokens = max_new_tokens # restore the original value of max_new_tokens

        model_output = f'{model_output}\n\n*************\n\n{reasoning}'
        prompt = f'{prompt_pt2}\n\n*************\n\n{prompt_pt1}'

        return prompt, model_output


    def judge_summary_quality_zeroshot(self, method, summary, judgment_type):
        prompt = self.prompts["Judge Code Summarization With Instructions"] if judgment_type == 'extended_instructions' \
            else self.prompts["Judge Code Summarization"]
        prompt = self.replace_tags(
            prompt = prompt,
            language = self.language.capitalize(),
            function = method,
            comment = summary
        )

        if 'CodeLlama-' in self.model_name and self.language == 'python':
            prompt = prompt + '\n\n# Rating:\nContent Adequacy:'
        else:
            prompt = prompt + '#'
        
        model_output = self.query_model(prompt = prompt)

        return prompt, model_output



    def judge_code_summary(self, method, summary, judgment_type):
        """
            Description of the Function:
                queries the model for the code summarization task on a given instance of the CoderEval dataset.

            Parameters:
                method (str): method to summarize;
                summary (str): candidate summary to evaluate;
                with_instructions (bool): whether to include detailed instructions about CA, Conciseness and Fluency in the prompt.
            
            Returns:
                str: prompt given to the model;
                str: output of the model.
        """

        if 'stepbystep' in judgment_type:
            prompt, model_output = self.judge_summary_quality_stepbystep(
                method = method,
                summary = summary,
                judgment_type = judgment_type
            )
        
        else:
            prompt, model_output = self.judge_summary_quality_zeroshot(
                method = method,
                summary = summary,
                judgment_type = judgment_type
            )

        return prompt, model_output