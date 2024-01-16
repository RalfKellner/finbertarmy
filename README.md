# How TO

This project is created with poetry. If you want to use this folder as is, go to https://python-poetry.org/ and follow the instructions for installation and usage.

# Project Plan

1. Evaluate financial texts, further include 8Ks, Financial News, maybe discard ESG reports
2. Build finetune tasks: Financial Sentiment, ESG disclosure, 8K abnormal event returns
3. Pre-trainining for with varying sequence length (252, 512), different number of hidden layers (6, 12) and maybe embedding dimension:
    - Roberta
    - Distilbert
    - Maybe Albert
    - Check Electra
4. Pre-training with financial masking
5. Develop pre-training evaluation task: accuracy for tokens and accuracy and for financial tokens in particular
5. Evaluate the relationship between pre-training, model complexity and the performance of fine-tuned models