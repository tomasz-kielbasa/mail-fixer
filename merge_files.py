
import os
import pandas as pd


def merge_files():
    directory = 'data/raw_mail'
    final_df = None

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        if os.path.isfile(filepath):
            df = pd.read_csv(filepath)
            df['source'] = filename[:-4]
            if final_df is None:
                final_df = df
            else:
                final_df = pd.concat((final_df, df), ignore_index=True)

    final_df.to_csv('data/merged_mail.csv', index=False)

    
def split_emails():
    new_mails = list()
    df = pd.read_csv('data/merged_mail.csv')
    for _, row in df.iterrows():
        email = {
            'email': row['email'],
            'author_name': row['author_name'],
            'recipient_name': row['recipient_name']
        }
        response = {
            'email': row['response'],
            'author_name': row['recipient_name'],
            'recipient_name': row['author_name']
        }
        new_mails.append(email)
        new_mails.append(response)
        
    split_df = pd.DataFrame(new_mails)
    split_df.to_csv('data/split_mail.csv', index=False)


if __name__ == '__main__':
    merge_files()
    split_emails()