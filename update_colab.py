import json

with open('colab_train.ipynb', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('"TICKER = \'AAPL\'          # Stock symbol to train on\\n"', '"TICKER = \'AAPL\' #@param {type:\\"string\\"}\\n"')
text = text.replace('"TIMEFRAME = \'1d\'         # Daily bars\\n"', '"TIMEFRAME = \'1d\' #@param [\\"1d\\", \\"1h\\", \\"15m\\"]\\n"')
text = text.replace('"TOTAL_STEPS = 500_000    # Training steps (500k ~ 30min on T4)\\n"', '"TOTAL_STEPS = 500000 #@param {type:\\"integer\\"}\\n"')

with open('colab_train.ipynb', 'w', encoding='utf-8') as f:
    f.write(text)
