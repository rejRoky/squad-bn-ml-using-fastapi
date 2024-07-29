### Dataset Summary
This is a Question Answering (QA) dataset for Bengali, curated from the SQuAD 2.0, TyDI-QA datasets and using the state-of-the-art English to Bengali translation model introduced here.

### Supported Tasks and Leaderboards
The dataset can be used for Question Answering tasks and can be used to train models for Bengali QA tasks.

### Languages
The text in the dataset is in Bengali.

## Dataset Structure
### Data Instances
A typical data point comprises a context, a question, and an answer. The context is a paragraph in Bengali, the question is a question in Bengali, and the answer is a span of text from the context.


### Data Fields
- `context`: a paragraph in Bengali
- `question`: a question in Bengali
- `answers`: a dictionary containing the following fields:
    - `text`: a span of text from the context
    - `answer_start`: the character offset of the start of the answer span in the context

### Data Splits
The dataset is split into a training set and a validation set.

## Dataset Creation
### Curation Rationale
The dataset was curated from the SQuAD 2.0, TyDI-QA datasets and using the state-of-the-art English to Bengali translation model introduced here.

### Source Data
The dataset was curated from the SQuAD 2.0, TyDI-QA datasets and using the state-of-the-art English to Bengali translation model introduced here.

#### Initial Data Collection and Normalization
The dataset was curated from the SQuAD 2.0, TyDI-QA datasets and using the state-of-the-art English to Bengali translation model introduced here.
