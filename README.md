# GPT-4-ENEM

**\*\*\* Most of the code in this repository was copied from the original 
[Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). \*\*\***

## Introduction

This repository contains code and data used in the paper "[Evaluating GPT-3.5 and GPT-4 Models on Brazilian University Admission Exams](https://arxiv.org/abs/2303.17003)". The study explores the capabilities of Language Models (LMs) in solving high-stakes multiple-choice tests, using the *[Exame Nacional do Ensino Médio (ENEM)](https://www.gov.br/inep/pt-br/areas-de-atuacao/avaliacao-e-exames-educacionais/enem)* as a case study. The ENEM is a multidisciplinary entrance examination widely adopted by Brazilian universities, which poses challenging tasks for LMs since its questions may span multiple fields of knowledge, requiring understanding of information from diverse domains.

The paper analyzes responses generated by GPT-3.5 and GPT-4 models for questions presented in the 2009-2017 exams, as well as for questions of the 2022 exam, which were made public after the training of the models completed. Furthermore, different prompt strategies were tested, including the use of Chain-of-Thought (CoT) prompts to generate explanations to answers.

On the 2022 edition, the best-performing model, GPT-4 with CoT, achieved an accuracy of 87%, largely surpassing GPT-3.5 by 11 points.


| Area                 | code-davinci-002   |          |          | gpt-3.5-turbo   |          |          | gpt-4          |          |          |
|----------------------|--------------------|--------------------|--------------------|-----------------|----------|----------|----------------|----------|----------|
|                      | zero-shot          | three-shot | three-shot with CoT | zero-shot | three-shot | three-shot with CoT | zero-shot | three-shot | three-shot with CoT |
| Languages and Codes  |        78.79       |   87.88   |   72.73   |      75.76      |   81.82   |   69.70   |      84.85     |   87.88   |   87.88   |
| Human Sciences       |        89.19       |   94.59   |   91.89   |      91.89      |   89.19   |   94.59   |      94.59     |   94.59   |   94.59   |
| Natural Sciences     |        69.23       |   61.54   |   53.85   |      73.08      |   84.62   |   65.38   |      84.62     |   76.92   |   88.46   |
| Mathematics          |        18.18       |   27.27   |   50.00   |      18.18      |   36.36   |   54.55   |      40.91     |   50.00   |   72.73   |
| **Total**            |      **68.64**     | **72.88** | **70.34** |    **69.49**    | **76.27** | **73.73** |    **79.66**   | **80.51** | **87.29** |

<p style="text-align: center;">Results on ENEM 2022. Questions that require image comprehension were removed.</p>
 
 We make available all explanations, targets and predictions generated for all experiments with the ENEM 2022 dataset in the [reports](reports/) folder.

## Data

We made available the [ENEM-2022 dataset](data/enem/2022.json), created by parsing questions and alternatives from the latest edition of the ENEM test. The dataset was structured and annotated with tags indicating the domain:

- **TU** - Text Understanding
- **IU** - Image Understanding
- **MR** - Mathematical Reasoning
- **CE** - Chemical Elements
- **ML** - Multilanguage

## Reproducing the results
To reproduce the experiments described in the paper, please follow the steps below:

### 1. Clone the repository:
```bash
git clone https://github.com/piresramon/gpt-4-enem.git
```

### 2. Install the required packages:
```bash
pip install -e .
```
### 3. Set the OPENAI API key:
Visit [openai](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key) to retrieve API keys and insert into your the env variable.
```bash
OPENAI_API_SECRET_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
### 4. Run the experiments:
```bash
# running 3-shot with CoT for chatgpt
python main.py \
    --model chatgpt \
    --model_args engine=gpt-3.5-turbo-0301 \
    --task enem_cot_2022 \
    --num_fewshot 3 \
    --description_dict_path description.json

# running 3-shot with CoT for gpt-4
python main.py \
    --model chatgpt \
    --model_args engine=gpt-4-0314 \
    --task enem_cot_2022 \
    --num_fewshot 3 \
    --description_dict_path description.json
```

We have four different tasks:
1. **enem**: Enem Challenge (2009-2017) without Chain-of-thought prompting.
2. **enem_cot**: Enem Challenge (2009-2017) with Chain-of-thought prompting.
3. **enem_2022**: Enem 2022 without Chain-of-thought prompting.
4. **enem_cot_2022**: Enem 2022 with Chain-of-thought prompting.

It is possible to use a different number of few-shot examples (maximum 3).

## Citation
If you use this code or data in your research, please cite the following paper:

```bibtex
@misc{nunes2023evaluating,
      title={Evaluating GPT-3.5 and GPT-4 Models on Brazilian University Admission Exams}, 
      author={Desnes Nunes and Ricardo Primi and Ramon Pires and Roberto Lotufo and Rodrigo Nogueira},
      year={2023},
      eprint={2303.17003},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

<!-- ## Contact
If you have any questions or comments, please feel free to contact us at pires.ramon@gmail.com. -->