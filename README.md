# Mail fixer

## Usage

Install dependencies
```
pip install -r requirements.txt
```

Generate synthetic mails
```
./run.sh
```

Train your model
```
python train.py
```


## Results

We finetuned opt-125m model. The produced output is coherent, rarely repeats, creative and conveys the same information as the original mail. You can check out the model on [Huggingface](https://huggingface.co/spaces/tomaszki/mail_fixer)


## Miscellaneous

Work inspired by [TinyStories](https://arxiv.org/abs/2305.07759) and [Textbooks](https://arxiv.org/abs/2309.05463).

We use [data-baby-names](https://github.com/hadley/data-baby-names) and [occupations](https://github.com/johnlsheridan/occupations) datasets