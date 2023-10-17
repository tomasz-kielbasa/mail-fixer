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