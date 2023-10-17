import pandas as pd
from vllm import LLM, SamplingParams

from utils import get_prompt

sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)
llm = LLM(model='mistralai/Mistral-7B-Instruct-v0.1')


def get_instruction(mail):
    return f'''Summarize the following email.
Email:
{mail}'''


def generate():
    df = pd.read_csv('data/split_mail.csv')
    mails = df['email'].values
    prompts = [get_prompt(get_instruction(mail)) for mail in mails]
    
    outputs = llm.generate(prompts, sampling_params)
    
    summaries = list()
    for output in outputs:
        generated_text = output.outputs[0].text
        summaries.append(generated_text.strip())
    
    
    df['summary'] = summaries
    df.to_csv('data/summaries.csv', index=False)

if __name__ == '__main__':
    generate()