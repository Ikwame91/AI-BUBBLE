# Token by Token

An interactive page that teaches how large language models work by running tiny, real models in the browser.

## Language

**Station**:
One numbered stop on the page (Tokenize, Embed, Attend, …) pairing an explanation with an interactive panel.
_Avoid_: section, lab, step (when you mean the whole stop)

**Corpus**:
The fixed training text every in-page model learns from.
_Avoid_: dataset, training data

**CharModel**:
The character-level next-character model trained live by gradient descent in the Pretrain station.
_Avoid_: bigram net, training lab

**Decoder**:
The step that turns a model's scores for the next token into one chosen token: temperature, top-k, top-p, greedy, then the dice roll.
_Avoid_: sampler, generation loop

**Feedback**:
The learning signals shown in the Align and Reason stations: reward-model weights, majority votes, group-relative advantages.
_Avoid_: RLHF math, reward stuff
