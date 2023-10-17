import pandas as pd
import random
from vllm import LLM, SamplingParams

from utils import get_prompt

sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=1024)
llm = LLM(model='mistralai/Mistral-7B-Instruct-v0.1')


instructions = ['''You're {author}. Write an email to {recipient} that can be summarized as following. Use only very basic vocabulary. Start email with greeting, don't include subject.
Summary:
{summary}''',
                '''You're {author} and you barely know English. Write an email to {recipient} that can be summarized as following. Use only very basic vocabulary. Start email with greeting, don't include subject.
Summary:
{summary}''',
                '''You're {author} and you barely know English. Write an email to {recipient} that can be summarized as following. Use only very basic vocabulary and make many grammar mistakes. Start email with greeting, don't include subject.
Summary:
{summary}''',
                '''You're {author} and you barely know English. Write an email to {recipient} that can be summarized as following. Make a lot of grammar mistakes. Start email with greeting, don't include subject.
Summary:
{summary}''',
                '''You're {author}. Write an email to {recipient} that can be summarized as following. Write the email so that it sounds like written by 7 year-old child. Start email with greeting, don't include subject.
Summary:
{summary}''',
]

def get_instruction(summary, author, recipient):
    format_dict = {
        'summary': summary,
        'author': author,
        'recipient': recipient,
    }
    return random.choice(instructions).format_map(format_dict)


def generate():
    df = pd.read_csv('data/summaries.csv')
    summaries = df['summary'].values
    authors = df['author_name'].values
    recipients = df['recipient_name'].values
    prompts = [get_prompt(get_instruction(
        summary, author, recipient
    )) for summary, author, recipient in zip(summaries, authors, recipients)]
    outputs = llm.generate(prompts, sampling_params)
    
    simple_mails = list()
    for output in outputs:
        generated_text = output.outputs[0].text
        simple_mails.append(generated_text.strip())
    
    df['simple_mail'] = simple_mails
    df.to_csv('data/simple_mail.csv', index=False)

if __name__ == '__main__':
    generate()