# GPT-4-ENEM

**\*\*\* Most of the code in this repository has been adapted from the original 
[Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). \*\*\***

## Introduction

This repository contains code and data used in the following papers:
- 
- Evaluating GPT-4's Vision Capabilities on Brazilian University Admission Exams (coming soon!). 
<!-- (https://arxiv.org/abs/2303.17003). -->
- [Evaluating GPT-3.5 and GPT-4 Models on Brazilian University Admission Exams](https://arxiv.org/abs/2303.17003).

This most recent study presents a comprehensive framework to evaluate language models on entrance exams, which incorporates both textual and visual elements. We evaluate the two most recent editions of *[Exame Nacional do Ensino Médio (ENEM)](https://www.gov.br/inep/pt-br/areas-de-atuacao/avaliacao-e-exames-educacionais/enem)*, the main standardized entrance examination adopted by Brazilian universities.

Our study not only reaffirms the capabilities of GPT-4 as the state of the art for handling complex multidisciplinary questions, but also pioneers in offering a realistic assessment of multimodal language models on Portuguese examinations.

One of the highlights is that text captions transcribing visual content outperform the direct use of images, suggesting that the vision model has room for improvement. Yet, despite improvements afforded by images or captions, mathematical questions remain a challenge for these state-of-the-art models.

Significant improvements are noticeable when incorporating either textual or visual representations of images, with the difference nearing 10 points, particularly when utilizing captions.

<table>
  <tr>
    <th rowspan="2">Area</th>
    <th colspan="3" style="text-align: center;">ENEM 2022</th>
    <th colspan="3" style="text-align: center;">ENEM 2023</th>
  </tr>
  <tr>
    <th>without images</th>
    <th>with images</th>
    <th>with captions</th>
    <th>without images</th>
    <th>with images</th>
    <th>with captions</th>
  </tr>
  <tr>
    <td>Languages and Codes</td>
    <td>73.33</td>
    <td>82.22</td>
    <td>84.44</td>
    <td>84.44</td>
    <td>86.67</td>
    <td>91.11</td>
  </tr>
  <tr>
    <td>Human Sciences</td>
    <td>88.89</td>
    <td>95.56</td>
    <td>95.56</td>
    <td>95.56</td>
    <td>100.00</td>
    <td>100.00</td>
  </tr>
  <tr>
    <td>Natural Sciences</td>
    <td>73.33</td>
    <td>77.78</td>
    <td>82.22</td>
    <td>86.67</td>
    <td>91.11</td>
    <td>93.33</td>
  </tr>
  <tr>
    <td>Mathematics</td>
    <td>54.55</td>
    <td>61.36</td>
    <td>61.36</td>
    <td>54.55</td>
    <td>65.91</td>
    <td>75.00</td>
  </tr>
  <tr>
    <td style="font-weight: bold;">Total</td>
    <td style="font-weight: bold;">72.63</td>
    <td style="font-weight: bold;">79.33</td>
    <td style="font-weight: bold;">81.01</td>
    <td style="font-weight: bold;">80.45</td>
    <td style="font-weight: bold;">86.03</td>
    <td style="font-weight: bold;">89.94</td>
  </tr>
</table>

<p style="text-align: center;">Results of GPT-4V on ENEM 2022 and ENEM 2023.</p>

The best-performing model was the GPT-4, that achieved an accuracy of 90.5% on ENEM 2023, using captions, largely surpassing the GPT-3.5 by 17 points


<!-- The study explores the capabilities of Language Models (LMs) in solving high-stakes multiple-choice tests, using the *[Exame Nacional do Ensino Médio (ENEM)](https://www.gov.br/inep/pt-br/areas-de-atuacao/avaliacao-e-exames-educacionais/enem)* as a case study. The ENEM is a multidisciplinary entrance examination widely adopted by Brazilian universities, which poses challenging tasks for LMs since its questions may span multiple fields of knowledge, requiring understanding of information from diverse domains.

The paper analyzes responses generated by GPT-3.5 and GPT-4 models for questions presented in the 2009-2017 exams, as well as for questions of the 2022 exam, which were made public after the training of the models completed. Furthermore, different prompt strategies were tested, including the use of Chain-of-Thought (CoT) prompts to generate explanations to answers.

On the 2022 edition, the best-performing model, GPT-4 with CoT, achieved an accuracy of 87%, largely surpassing GPT-3.5 by 11 points. -->

<!-- | Area                 | code-davinci-002   |          |          | gpt-3.5-turbo   |          |          | gpt-4          |          |          |
|----------------------|--------------------|--------------------|--------------------|-----------------|----------|----------|----------------|----------|----------|
|                      | zero-shot          | three-shot | three-shot with CoT | zero-shot | three-shot | three-shot with CoT | zero-shot | three-shot | three-shot with CoT |
| Languages and Codes  |        78.79       |   87.88   |   72.73   |      75.76      |   81.82   |   69.70   |      84.85     |   87.88   |   87.88   |
| Human Sciences       |        89.19       |   94.59   |   91.89   |      91.89      |   89.19   |   94.59   |      94.59     |   94.59   |   94.59   |
| Natural Sciences     |        69.23       |   61.54   |   53.85   |      73.08      |   84.62   |   65.38   |      84.62     |   76.92   |   88.46   |
| Mathematics          |        18.18       |   27.27   |   50.00   |      18.18      |   36.36   |   54.55   |      40.91     |   50.00   |   72.73   |
| **Total**            |      **68.64**     | **72.88** | **70.34** |    **69.49**    | **76.27** | **73.73** |    **79.66**   | **80.51** | **87.29** |

<p style="text-align: center;">Results on ENEM 2022. Questions that require image comprehension were removed.</p>
 
 We make available all explanations, targets and predictions generated for all experiments with the ENEM 2022 dataset in the [reports](reports/) folder. -->

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