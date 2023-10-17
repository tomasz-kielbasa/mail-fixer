def get_prompt(text, start=None):
    if start is None:
        return f'[INST] {text} [/INST]'
    return f'[INST] {text} [/INST] {start}'