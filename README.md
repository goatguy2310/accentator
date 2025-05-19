# Accentator

An AI system that restores accents/tone marks from non-accent Vietnamese text using Transformer.

## Introduction

In the Vietnamese language, words can have multiple accents compared to English. For example, a single word "kho" can have different meanings based on the accent placed on it, i.e. "khó" means "hard", "khổ" means miserable, "khò" is the sleeping sound, even "kho" has a meaning which is storage. Because of this, multiple Vietnamese texts are without accents as it is much quicker to do so without the additionally hassle to click another or two buttons for that accent that people would probably understand anyway. However, there are cases where it can cause confusion, the case above is an example for that. Our project here is proposing a rough solution for that, by using AI and machine learning to turn non-accented text into accented one with its context.

Although limited, there has been studies on this problem in the past, with Transformer. Most notable is [duongntbk's repo](https://github.com/duongntbk/restore_vietnamese_diacritics), which incorporates the BERT architecture (Bidirectional Encoder Representations from Transformers) and see this as a machine translation problem. His method achieved 94.05% accuracy on test datasets. Another research related to this is from [Phuong](https://www.sciencedirect.com/science/article/pii/S0950705121007668), who also used the BERT architecture to generate diacritics from text online with the purpose of detecting hate speech on Vietnamese social media. They achieved around 92% accuracy.

In our project Accentator, we will apply a lightweight version of the GPT-2 structure, which is unidirectional and decoder-only, and see if it improves over bidirectional methods.

## Methodology

The structure of Accentator goes as follow:
- Word/Character embedding + Positional encoding
- 6 blocks of the module:
  - Layer Norm
  - Attention Layer
  - Layer Norm
  - Linear Layer + GeLU
  - Linear Layer
- Linear Layer to Output

Currently, the hyperparameters are:
- Context size: 128
- Vocab size:
- Head number: 6
- Embedding size: 256

We ran experiments on an NVIDIA A5000 24GB VRAM.

## Results
