# Task

(A-letter is accented letter, NA-letter is non-accented)

## Data processing
- Find a data source, i.e. the data used to train PhoGPT (https://github.com/VinAIResearch/PhoGPT) and find out how it's structured.
- Make a python script that transforms the data to the format `x = (window size of A-letters) <SEP> NA-letter`, `y = A-letter`. Of course each letter is tokenized to a number, but here I put it like this only for simplicity (for example a = 1, b = 2, à = 34, <SEP> = 50). For example: if the window size is 4, from the sentence "Mẹ mày béo" we can have `x = Mẹ m <SEP> a` `y = à`, `x = ày b <SEP> e` `y = é`. The window size can be larger of course. In the future, we might expand this to word-level prediction.

Useful links:
- https://github.com/VinAIResearch/PhoGPT
- https://github.com/karpathy/nanoGPT/tree/master
- https://www.youtube.com/watch?v=kCc8FmEb1nY
