import random
import json
import pandas as pd
from vllm import LLM, SamplingParams

from utils import get_prompt

sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=1024)
llm = LLM(model='mistralai/Mistral-7B-Instruct-v0.1')

def load_json(filename):
    with open('data/external_fixed/' + filename + '.json', 'r') as file:
        return json.load(file)

names = load_json('names')
professions = load_json('professions')
words = load_json('words')
mail_types = ['a very short email', 'a short message', 'an email']
email_starts = ['Hi', 'Hello', 'Dear']


def generate(prompt, response_prompt, name, n=1):
    prompts = list()
    data = list()
    for _ in range(n):
        example = {
            'author_name': random.choice(names),
            'profession': random.choice(professions),
            'recipient_name': random.choice(names),
            'word': random.choice(words),
            'mail_type': random.choice(mail_types),
            'start': random.choice(email_starts),
            'start_response': random.choice(email_starts),
        }
        data.append(example)
        prompts.append(get_prompt(prompt.format_map(example), example['start']))
    outputs = llm.generate(prompts, sampling_params)
    
    for output, example in zip(outputs, data):
        generated_text = output.outputs[0].text
        email_text = example['start'] + generated_text
        example['email'] = email_text
    prompts = [get_prompt(response_prompt.format_map(example), example['start_response']) for example in data]
    outputs = llm.generate(prompts, sampling_params)
    
    for output, example in zip(outputs, data):
        generated_text = output.outputs[0].text
        email_text = example['start_response'] + generated_text
        example['response'] = email_text
    
    df = pd.DataFrame(data, columns=[
        'email', 'response', 'author_name', 'recipient_name', 'profession'
    ])
    df.to_csv('data/raw_mail/' + name + '.csv', index=False)

if __name__ == '__main__':
    data_per_task = 20_000
    # Employee to boss
    prompt = "You're {author_name}, a {profession}. Write {mail_type} to your boss {recipient_name}. " \
             "Write about anything you want. You should make up facts and names. Use word \"{word}\". " \
             "Don't start email with \"I hope this email finds you well\"."
    response_prompt = "You're {recipient_name}, a boss. " \
                      "You received an email from your employee {author_name}, who works as a {profession}. " \
                      "You should make up facts. Answer the following email. Email:\n{email}"
    generate(prompt, response_prompt, 'employee_to_boss', data_per_task)
    
    # Boss to employee
    prompt = "You're {author_name}, a boss. Write {mail_type} to your employee {recipient_name}, who works as a {profession}. " \
             "Write about anything you want. You should make up facts and names. Use word \"{word}\". " \
             "Don't start email with \"I hope this email finds you well\"."
    response_prompt = "You're {recipient_name}, a {profession}. " \
                      "You received an email from your boss {author_name}. " \
                      "You should make up facts. Answer the following email. Email:\n{email}"
    generate(prompt, response_prompt, 'boss_to_employee', data_per_task)
             
    # Employee to employee
    prompt = "You're {author_name}. Write {mail_type} to your colleague {recipient_name}, you both work as a {profession}. " \
             "Write about anything you want. You should make up facts and names. Use word \"{word}\". " \
             "Don't start email with \"I hope this email finds you well\"."
    response_prompt = "You're {recipient_name} " \
                      "You received an email from your colleague {author_name}, you both work as a {profession} " \
                      "You should make up facts. Answer the following email. Email:\n{email}"
    generate(prompt, response_prompt, 'employee_to_employee', data_per_task)
    
    # Candidate to recruiter
    prompt = "You're {author_name}. Write {mail_type} to a recruiter {recipient_name}. Inquire about {profession} position. " \
             "Write about anything you want, ask questions about the role. You should make up facts and names. Use word \"{word}\". " \
             "Don't start email with \"I hope this email finds you well\"."
    response_prompt = "You're {recipient_name}, a recruiter. " \
                      "You received an email from from a candidate {author_name}, who asks about {profession} position." \
                      "You should make up facts. Answer the following email. Email:\n{email}"
    generate(prompt, response_prompt, 'candidate_to_recruiter', data_per_task)
    
    # Recruiter to candidate
    prompt = "You're {author_name}, a recruiter. Write {mail_type} to a candidate {recipient_name} for {profession} position. " \
             "Write about anything you want, you can schedule interview. You should make up facts and names and stage of recruitment process. Use word \"{word}\". " \
             "Don't start email with \"I hope this email finds you well\"."
    response_prompt = "You're {recipient_name} " \
                      "You received an email from from a recruiter {author_name}." \
                      "You should make up facts. Answer the following email. Email:\n{email}"
    generate(prompt, response_prompt, 'recruiter_to_candidate', data_per_task)
    
    # To client
    prompt = "You're {author_name}, a {profession}. Write {mail_type} to your client {recipient_name}. " \
             "Write about anything you want. You should make up facts and names. Use word \"{word}\". " \
             "Don't start email with \"I hope this email finds you well\"."
    response_prompt = "You're {recipient_name}." \
                      "You received an email from {author_name}, who works as {profession}. " \
                      "You should make up facts. Answer the following email. Email:\n{email}"
    generate(prompt, response_prompt, 'to_client', data_per_task)
    