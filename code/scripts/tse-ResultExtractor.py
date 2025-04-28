import json
import pandas as pd
from datetime import datetime
import numpy as np
import lizard
import re

class ResultExtractor:
    def __init__(self, debug_mode='../'):
        self.debug_mode = debug_mode
        self.constants = self._load_json(f'{debug_mode}constants/constants_json.json')
        self.iep_models = self._load_json(f'{debug_mode}constants/IEPmodels.json')
        self.iep_headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.constants['Hugging Face Inference Endpoints Token']}",
            "Content-Type": "application/json"
        }

    @staticmethod
    def _load_json(file_path : str):
        if file_path.split('.')[-1] == 'json':
            return json.load(open(file_path))
        
        elif file_path.split('.')[-1] == 'jsonl':
            data = []
            with open(file_path, 'r') as file:
                for line in file:
                    data.append(json.loads(line))
            return data

    @staticmethod
    def _extract_generatedby_from_multiple_filename(input_string : str):
        if 'emptymethods' in input_string or 'trivialmethods' in input_string:
            return input_string
        elif 'gpt-' in input_string:
            model_parts = []
            for part in input_string.split('-'):
                if part == 'gpt':
                    model_parts.append(part)
                elif model_parts:
                    if part in {'turbo', '4o'}:
                        model_parts.append(part)
                        break
                    model_parts.append(part)

            return "-".join(model_parts)
        else:
            input_string = input_string.split('-')[2]
            prep = [x for x in ['codellama_', 'deepseek_ai_', 'Qwen_', 'microsoft_', 'bigcode_'] if x in input_string][0]
            model_name = input_string.split(prep)[-1]
            model_name = model_name.replace('_', '-')
            return model_name
    
    @staticmethod
    def _replace_quotes_with_placeholders(input_string : str):
        return input_string.replace("\"(\"", "<DOUBLE_QUOTE_OPENING_ROUND>")\
                        .replace("\")\"", "<DOUBLE_QUOTE_CLOSING_ROUND>")\
                        .replace("\"[\"", "<DOUBLE_QUOTE_OPENING_SQUARE>")\
                        .replace("\"]\"", "<DOUBLE_QUOTE_CLOSING_SQUARE>")\
                        .replace("\"{\"", "<DOUBLE_QUOTE_OPENING_CURLY>")\
                        .replace("\"}\"", "<DOUBLE_QUOTE_CLOSING_CURLY>")\
                        .replace("'('", "<SINGLE_QUOTE_OPENING_ROUND>")\
                        .replace("')'", "<SINGLE_QUOTE_CLOSING_ROUND>")\
                        .replace("'['", "<SINGLE_QUOTE_OPENING_SQUARE>")\
                        .replace("']'", "<SINGLE_QUOTE_CLOSING_SQUARE>")\
                        .replace("'{'", "<SINGLE_QUOTE_OPENING_CURLY>")\
                        .replace("'}'", "<SINGLE_QUOTE_CLOSING_CURLY>")\
                        .replace("'", "<SINGLE_QUOTE>")\
                        .replace("\"", "<DOUBLE_QUOTE>")
    
    @staticmethod
    def _replace_placeholders_with_quotes(input_string : str):
        return input_string.replace("<DOUBLE_QUOTE_OPENING_ROUND>", "\"(\"")\
                .replace("<DOUBLE_QUOTE_CLOSING_ROUND>", "\")\"")\
                .replace("<DOUBLE_QUOTE_OPENING_SQUARE>", "\"[\"")\
                .replace("<DOUBLE_QUOTE_CLOSING_SQUARE>", "\"]\"")\
                .replace("<DOUBLE_QUOTE_OPENING_CURLY>", "\"{\"")\
                .replace("<DOUBLE_QUOTE_CLOSING_CURLY>", "\"}\"")\
                .replace("<SINGLE_QUOTE_OPENING_ROUND>", "'('")\
                .replace("<SINGLE_QUOTE_CLOSING_ROUND>", "')'")\
                .replace("<SINGLE_QUOTE_OPENING_SQUARE>", "'['")\
                .replace("<SINGLE_QUOTE_CLOSING_SQUARE>", "']'")\
                .replace("<SINGLE_QUOTE_OPENING_CURLY>", "'{'")\
                .replace("<SINGLE_QUOTE_CLOSING_CURLY>", "'}'")\
                .replace("<SINGLE_QUOTE>", "'")\
                .replace("<DOUBLE_QUOTE>", "\"")
    
    @staticmethod
    def _extract_docstring_from_multiple_test_output(row):
        input_program = row['program']
        signature = row['signature']
        docstring = ''

        for line in input_program.split('\n'):
            if signature in line:
                break
            if line.strip().startswith('//'):
                docstring += f'{line.strip()}\n'

        return docstring.strip()
    
    
    
    @staticmethod
    def _search_java_function(file_path : str):
        """
        Description of the Function:
            returns the Start line and End line of all the java methods found in a given input file.

        Parameters:
            file_path (str): path to the file to read.

        Returns:
            list of lists [N x 3]: for each java method found, it returns a list containing the Name of the method, the Start line and the End line of the method.
    """
        
        liz = lizard.analyze_file(file_path)
        functions_info = []
        for liz_elem in liz.function_list:
            functions_info.append([liz_elem.long_name, liz_elem.start_line, liz_elem.end_line])
        
        return functions_info

    
    def extract_method_from_multiple_test_output(self, row):
        # return as method what is in between the signature (included) and the last occurrence of the end_pattern (the beginning of tests automatically added by multiple)
        start_pattern = row['signature']
        end_pattern = "public static void main(String[] args) {"

        regex = re.compile(f"({re.escape(start_pattern)}.*)(?={re.escape(end_pattern)}(?!.*{re.escape(end_pattern)}))", re.DOTALL)
        match = regex.search(row['program'])

        if match:
            method = match.group(1).strip()
            return method
        else:
            return ''
    
    
    def extract_predicted_method_from_output(self, model_output, file_extention, gpt_flag = False, hat = '', method_index = 0, tempfile_name = 'tempFile'):
        """
        Description of the Function:
            returns all the methods found in the output of a model.

        Parameters:
            model_output (str): output of the model in which to search for java methods;
            hat (str): optional "hat" to prepend to the model output;
            method_index (int or array): index of the method that wants to be extracted (ie, if [0,1] the first and the second method are extracted).

        Returns:
            list: list of the java methods extracted. If no java methods are present in the model output, an empty list is returned.
        """
        methods_found = []
        
        # if the call to the model failed
        if 'FAILED.' in model_output or len(model_output) == 0:
            return '' #if multiple else methods_found
        
        model_output = self._replace_quotes_with_placeholders(model_output)

        # write the model output to a .java/.py file
        prediction_file = '{0}data/temp/{1}{2}.{3}'.format(self.debug_mode, tempfile_name.split('/')[-1], datetime.now(), file_extention)
        with open(prediction_file, 'w') as w:
            w.write(hat)
            w.write('\n') if model_output[0] != '\n' else w.write('') # models are expected to continue writing the function starting from the signature, so the first char must be a newline
            w.write('    ') if file_extention == 'py' and not gpt_flag else w.write('') # if the target is a python method the first line after the signature is indentated
            
            # special attention to methods with annotation
            was_hat = True if len(hat) > 0 and hat[0] == '@' else False
            for line in model_output.split('\n'):
                w.write('{}\n'.format(line.strip(' '))) if was_hat else w.write(f'{line}\n') # the line following the one of the annotation is stripped
                was_hat = True if len(line) > 0 and line [0] == '@' else False

        w.close()

        listMethods = self._search_java_function(prediction_file) # run the parser on the written file
        
        with open(prediction_file, 'r') as r:
            # loop over the methods that were found by the parser
            for method in listMethods:
                methodStartLine, methodEndLine = int(method[1]) - 1, int(method[2]) - 1
                method_to_append = ''
                r.seek(0)
                for i, line in enumerate(r):
                    line = self._replace_placeholders_with_quotes(line)
                    
                    # check the line before the start line of the method
                    if i == methodStartLine - 1 and len(line.strip('\n\t \"\'')) > 0 and line.strip('\n\t \"\'')[0] == '@': #if the line before the method signature there is an annotation
                        annotation = line.strip('\n')
                        method_to_append += f'{annotation}\n' if file_extention == 'py' else f'{annotation} '
                        
                    # append the lines of the method
                    if i >= methodStartLine and i <= methodEndLine:
                        method_to_append += f'{line}'
                
                methods_found.append(method_to_append.strip('\n\t \"\''))
        r.close()
            
        methods_found = np.array(methods_found)

        try:
            # this covers the case in which a function is defined inside another function and we want to extract the external one
            if int(listMethods[method_index + 1][1] - 1) < int(listMethods[method_index][1] - 1) and int(listMethods[method_index + 1][2] - 1) > int(listMethods[method_index][2] - 1):
                return methods_found[method_index + 1]
            return methods_found[method_index]
        except IndexError:
            try:
                #goes here in the case where the method_index+1 is out of bounds but method_index is not
                return methods_found[method_index]
            except IndexError:
                return '' #if multiple else methods_found


    def extract_results_from_multiple_test_output(self, file_path):
        """
        Description of the Function:
            parses the files produced by the test suites of the MultiPL-E dataset and returns a dataframe with the results of the tests.

        Parameters:
            file_path (str): .json file containing the output of the tests;

        Returns:
            DataFrame with columns
                -> dataset (str): the dataset to which the candidate method belongs to (ie, humaneval, mbpp ...);
                -> target_id (str): the id of the code generation problem for which the candidate was generated;
                -> description (str): the method description given as a input to the LLM which generated the candidate;
                -> signature (str): the signature of the candidate;
                -> method (str): the candidate implementation automatically generated;
                -> generated_by (str): the automatic technique which generated the candidate;
                -> exit_code (int): 0 if the candidate is correct (ie, passes the tests) or 1 if it is incorrect.
        """
        test_output_list = self._load_json(file_path)
        input_path = file_path.split('/')[-2]

        test_output_dict = {}
        for result in test_output_list['results']:
            for k, v in result.items():
                test_output_dict.setdefault(k, []).append(v)

        test_output_df = pd.DataFrame(test_output_dict)
        test_output_df['dataset'] = '-'.join(input_path.split('-')[0:2])
        test_output_df['target_id'] = test_output_list['name']
        test_output_df['generated_by'] = self._extract_generatedby_from_multiple_filename(input_path)
        test_output_df['signature'] = test_output_list['prompt'].split('\n')[-2].strip()
        test_output_df['method'] = test_output_df.apply(self.extract_method_from_multiple_test_output, axis = 1)
        # test_output_df['method'] = test_output_df['program'].apply(lambda x : x)
        test_output_df['description'] = test_output_df.apply(self._extract_docstring_from_multiple_test_output, axis = 1)

        return test_output_df[['dataset', 'target_id', 'description', 'signature', 'method', 'generated_by', 'exit_code']]
    

    def extract_ispass_from_codereval_test_output(self, file_path):
        """
        Description of the Function:
            parses the files produced by the test suites of the CoderEval dataset and returns a dataframe with the results of the tests.

        Parameters:
            file_path (str): .jsonl file containing the output of the tests;

        Returns:
            DataFrame with columns
                -> target_id (str): the id of the code generation problem for which the candidate was generated;
                -> generated_code (str): the candidate implementation automatically generated;
                -> is_pass (int): 0 if the candidate is wrong (ie, fails the tests) or 1 if it is correct.
        """
        data = self._load_json(file_path)
        df = pd.DataFrame(data)
        method_id = [df._id.iloc[i] for i in range(df.shape[0]) for _ in range(len(df.generate_results.iloc[i]))]
        is_pass = [df.generate_results.iloc[i][j]['is_pass'] for i in range(df.shape[0]) for j in range(len(df.generate_results.iloc[i]))]
        generated_code = [df.generate_results.iloc[i][j]['generate_code'] for i in range(df.shape[0]) for j in range(len(df.generate_results.iloc[i]))]

        return pd.DataFrame({
            'target_id': method_id,
            'generated_code': generated_code,
            'is_pass': is_pass
            })
    
    
    def extract_predictions_for_codereval(self, file_path, batch_ids = None):
        """
        Description of the Function:
            given a path to a jsonl file containing predictions of the CoderEval dataset, 
            it returns a dataframe containing the target_id and the candidate implementation. 
            It only returns the instances specified in the batch_ids list.
            This function handles the case in which to each target was associated only one candidate (beam = 1).

        Parameters:
            file_path (str): path to the jsonl file which contains a set of predictions of the CoderEval dataset;
            batch_ids (list) : contains a subset of the target ids of the CoderEval dataset;

        Returns:
            pandas.DataFrame: dataframe with the following columns
                -> target_id (str): id of the instance;
                -> generated_code (str): a candidate function;
    """
        results = {}
        with open(file_path, 'r') as file:
            for line in file:
                json_obj = json.loads(line)
                tid = json_obj['_id']
                if batch_ids and tid not in batch_ids:
                    continue
                results.setdefault('target_id', []).append(tid)
                results.setdefault('generated_code', [])
                results['generated_code'].append(json_obj['generate_results'][0]) \
                    if len(json_obj['generate_results']) > 0 else results['generated_code'].append('')

        return pd.DataFrame(results)






    def remove_assert_from_java_method(self, method_implementation):

        # write the original implementation from the second line to the second to last
        with open('/home/giuseppe/law_school/data/temp/wo_first_last.java', 'w') as w:
            original_lines = method_implementation.splitlines()
            for line in original_lines[1:-1]:
                line = self._replace_quotes_with_placeholders(line)
                w.write(f'{line}\n')
        w.close()

        # look for internal java methods
        method_list = self._search_java_function('/home/giuseppe/law_school/data/temp/wo_first_last.java')

        found_assert = False
        for method in method_list:
            start_line, end_line = int(method[1]) - 1, int(method[2]) - 1
            with open('/home/giuseppe/law_school/data/temp/wo_first_last.java', 'r') as r:
                for line_number, line in enumerate(r):
                    if line_number == start_line:
                        line = self._replace_placeholders_with_quotes(line)

                        if line.strip().startswith('public static void main'): # if finds an internal main function
                            found_assert = True
                            break
            r.close()
        
        if not found_assert:
            return method_implementation

        # retrieves all the lines but the ones of the internal main function
        lines_toretrieve = []
        for iline, line in enumerate(original_lines):
            if not iline in np.arange(start_line + 1, end_line + 2):
                lines_toretrieve.append(line)
        
        return '\n'.join(lines_toretrieve)
    