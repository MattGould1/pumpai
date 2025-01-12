# pumpai

## Important

THIS IS NOT TRADE ADVICE! This is intended to be educational only.

Never click links in this repository leaving github, never click links in Issues, don't run code that others post without reading it, this software is provided "as is," without warranty.

## Overview

PumpAI is a tensorflow project aimed to generate `buy`, `sell` or `wait` signals based on trades created from `pump.fun`.

In the `data/mints` directory you will find:

- `all_mint.csv` which is a CSV file containing a few hours of trade data from pump.fun
- `mint_train.csv` which is data that has been labeled in `mint_label.csv` and is grouped into ragged tensors later
- `mint_label.csv` which are labels for `mint_train.csv`

I have zero expertise in understanding trade signals, and so my labelling is very likely of poor quality. This means, that although educational for me, the project will likely only generate accurate predictions for my own labelling that is unlikely to apply broadly.

Most tokens on pump.fun are short lived, becoming stale very quickly, with a minority picking up traction

The life of tokens on pump.fun are usually short lived with many becoming stal

The data received from pump.fun is usually short lived with many new tokens generated. This creates ragged data like `[batch_size, None, 9]`. The `None` are the trades themselves which for any given token can be an arbitrary length. Keras (https://github.com/tensorflow/tensorflow/issues/65399) currently does not support ragged data very well, and so I normalize the ragged data and fill with zeroes.

The goal of this project is to achieve a high accuracy when evaluating test data. DO NOT USE THIS FOR TRADING ON PUMP FUN!

## Usage

```bash
$ python -m pumpai
#or
$ pumpai
```

## Disclaimer

This software is provided "as is," without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

Use at your own risk. The authors take no responsibility for any harm or damage caused by the use of this software. Users are responsible for ensuring the suitability and safety of this software for their specific use cases.

By using this software, you acknowledge that you have read, understood, and agree to this disclaimer.
