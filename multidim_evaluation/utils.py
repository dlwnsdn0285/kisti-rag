import os, json, sys, re

def extract_json_from_response(response):
    if not response:
        return response
    if isinstance(response, dict) and 'content' in response:
        response = response['content']
    
    solution_pattern = r'<solution>(.*?)</solution>'
    solution_match = re.search(solution_pattern, response, re.DOTALL)
    if solution_match:
        solution_content = solution_match.group(1).strip()
        if '"Answer":' in solution_content:
            json_pattern = r'\{\s*"Answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"[^}]*\}'
            json_matches = re.findall(json_pattern, solution_content)
            if json_matches:
                last_json_answer = json_matches[-1]
                return last_json_answer.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\u202f', ' ')
        
            answer_match = re.search(r'"Answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', solution_content)
            if answer_match:
                return answer_match.group(1).replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\u202f', ' ')
    
        boxed_pattern = r'\\boxed\{(.*?)\}'
        boxed_match = re.search(boxed_pattern, solution_content, re.DOTALL)
        if boxed_match:
            boxed_content = boxed_match.group(1).strip()
            text_pattern = r'\\text\{(.*?)\}'
            text_match = re.search(text_pattern, boxed_content, re.DOTALL)
            if text_match:
                return text_match.group(1).strip()
            return boxed_content
        return solution_content.strip()
    
    if '"Answer":' in response:
        json_pattern = r'\{\s*"Answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"[^}]*\}'
        json_matches = re.findall(json_pattern, response)
        if json_matches:
            last_json_answer = json_matches[-1]
            return last_json_answer.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\u202f', ' ')
        last_answer_index = response.rfind('"Answer":')
        if last_answer_index == -1:
            return str(response)
        answer_section = response[last_answer_index:]
        try:
            answer_match = re.search(r'"Answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', answer_section)
            if answer_match:
                return answer_match.group(1).replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\u202f', ' ')
        except:
            pass
        parts = answer_section.split('"Answer":', 1)
        if len(parts) > 1:
            answer_part = parts[1].strip()
            if answer_part.startswith(':'):
                answer_part = answer_part[1:].strip()
            if answer_part.startswith('"'):
                answer_part = answer_part[1:]
            if answer_part.endswith('"}'):
                answer_part = answer_part[:-2]
            elif answer_part.endswith('"'):
                answer_part = answer_part[:-1]
            elif answer_part.endswith('}'):
                if answer_part.endswith('"}'):
                    answer_part = answer_part[:-2]
                else:
                    answer_part = answer_part[:-1]
                if answer_part.endswith('"'):
                    answer_part = answer_part[:-1]
            while answer_part.endswith('"'):
                answer_part = answer_part[:-1]
            return str(answer_part)
    return str(response)