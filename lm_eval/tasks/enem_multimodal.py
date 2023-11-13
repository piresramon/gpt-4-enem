"""
Evaluating GPT-3.5 and GPT-4 Models on Brazilian University Admission Exams
https://arxiv.org/abs/2303.17003

The study explores the capabilities of Language Models (LMs) in solving 
high-stakes multiple-choice tests, using the Exame Nacional do Ensino Médio 
(ENEM) as a case study. The ENEM is a multidisciplinary entrance examination 
widely adopted by Brazilian universities, which poses challenging tasks for 
LMs since its questions may span multiple fields of knowledge, requiring 
understanding of information from diverse domains.

Homepage: https://github.com/piresramon/gpt-4-enem
"""
import collections
import json
import numpy as np
import os

from lm_eval import utils
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from lm_eval.tasks.enem import ENEM

_CITATION = """
@misc{nunes2023evaluating,
      title={Evaluating GPT-3.5 and GPT-4 Models on Brazilian University Admission Exams}, 
      author={Desnes Nunes and Ricardo Primi and Ramon Pires and Roberto Lotufo and Rodrigo Nogueira},
      year={2023},
      eprint={2303.17003},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

class ENEM_2022(ENEM):
    VERSION = 0
    DATASET_PATH = 'data/enem'
    DATASET_NAME = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):

        self.dataset = collections.defaultdict(list)
        
        fname = os.path.join(self.DATASET_PATH, '2022.jsonl')
        with open(fname, 'r', encoding='utf-8') as f:
            documents = [json.loads(line) for line in f]

        documents = [d for d in documents if d['label'] in ['A', 'B', 'C', 'D', 'E']] # remove questions annulled?

        # Experimento com descrição: deixa o dataset completo
        # Experimento com imagem: tira a descrição
        # Experimento com nada: tira descrição e imagem

        for d in documents: # TEMP remove description to use the image
            d['description'] = ''
        # for d in documents: # TEMP remove figures
        #     d['figures'] = []
        

        # TODO: prepare method to select vision, ledor, or blind        
        # def ignore_question(doc):
        #     filters = {
        #         'IU': False,
        #         # 'MR': False,  # uncomment to filter out MR
        #         # 'CE': False,  # uncomment to filter out CE
        #         'ML': False,
        #     }
        #     for k,v in filters.items():
        #         if doc[k] != v:
        #             return True
        #     return False

        # documents = list(filter(lambda doc: not ignore_question(doc), documents))
        self.dataset['test'] = list(map(self._process_doc, documents))

    def process_results(self, doc, results):
        results = super().process_results(doc, results)

        q_id = int(doc['id'].split('_')[-1])
        area = ['languages', 'human-sciences', 'natural-sciences', 'mathematics'][int(np.ceil(q_id/45))-1]

        results[area] = results['acc']
        # results['c_' + area] = 1  # just to count number of questions per area
        return results
    
    def _process_doc(self, doc):
        def format_example(doc, choices):
            """
                Passagem: <passage>
                Pergunta: <question>
                Choices:
                A. <choice1>
                B. <choice2>
                C. <choice3>
                D. <choice4>
                Answer:
            """
            prompt = doc.get('context', "") + '\n' + doc.get("question", "")
            prompt = prompt.strip() + '\n'
            alternatives = doc.get('alternatives', doc.get('options'))
            for choice, option in zip(choices, alternatives):
                prompt += f"{choice.upper()}. {option}\n"
            prompt += "Resposta:"
            return prompt
        choices = ['A', 'B', 'C', 'D', 'E']
        return {
            "query": format_example(doc, choices),
            "choices": doc.get('alternatives', doc.get('options')),
            "gold": choices.index(doc["label"].upper()),
            "id": doc["id"],
            "exam": doc["exam"],
            "description": doc.get("description", ""),
            "figures": doc.get("figures", []),
        }
 
    def test_docs(self):
        return self.dataset['test']

    def higher_is_better(self):
        return {
            "acc": True,
            '2022': True,
            'languages': True,
            'human-sciences': True,
            'natural-sciences': True,
            'mathematics': True,
            'c_languages': True,
            'c_human-sciences': True,
            'c_natural-sciences': True,
            'c_mathematics': True,
        }
    
    def aggregation(self):
        return {
            "acc": mean,
            '2022': mean,
            'languages': mean,
            'human-sciences': mean,
            'natural-sciences': mean,
            'mathematics': mean,
            'c_languages': sum,
            'c_human-sciences': sum,
            'c_natural-sciences': sum,
            'c_mathematics': sum,
        }

    @utils.positional_deprecated
    def fewshot_context(self, doc, num_fewshot, provide_description=None, rnd=None, description=None):
        """ Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param prompt_mode: str
            The type of prompt. Please set prompt_mode as "fixed", "dynamic-random", or "dynamic-similar".
            WARNING: this is implemented only for Portuguese tasks.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert rnd is not None, "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print("WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict")

        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                # fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
                ## keeping the training docs in original order (use this to fixed prompts)
                fewshotex = list(self.training_docs())[:num_fewshot]
                ## if the current doc is among the training docs, we do not use it as few-shot
                fewshotex = [ex for ex in fewshotex if doc['id'] != ex['id']]
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs() if self.has_validation_docs() else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = ''
            for i, doc_ex in enumerate(fewshotex):
                labeled_examples += f'Questão {i+1}:\n'
                labeled_examples += self.doc_to_text(doc_ex) + self.doc_to_target(doc_ex)
                labeled_examples += '\n##\n'
            labeled_examples += f'Questão {len(fewshotex) + 1}:\n'

        example = self.doc_to_text(doc)
        return description + labeled_examples + example


# class ENEM_MULTIMODAL_CoT(ENEM_MULTIMODAL):

#     def _process_doc(self, doc):
#         def format_example(doc, choices):
#             """
#                 Passagem: <passage>
#                 Pergunta: <question>
#                 Choices:
#                 A. <choice1>
#                 B. <choice2>
#                 C. <choice3>
#                 D. <choice4>
#                 Answer:
#             """
#             prompt = "Cabeçalho: " + doc["context"] + "\n"
#             prompt += "Enunciado: " + doc["question"] + "\nAlternativas:\n"
#             for choice, option in zip(choices, doc["options"]):
#                 prompt += f"{choice.upper()}. {option}\n"
            
#             prompt += "Explicação: " + doc.get("explanation", "")
#             return prompt.strip()
#         choices = ['a', 'b', 'c', 'd', 'e']
#         return {
#             "query": format_example(doc, choices),
#             "choices": doc["options"],
#             "gold": choices.index(doc["label"]),
#             "id": doc["id"],
#             "exam": doc["exam"],
#         }
    
#     def doc_to_target(self, doc):
#         return " " + ['A.', 'B.', 'C.', 'D.', 'E.'][doc['gold']].upper()

#     def construct_requests(self, doc, ctx):
#         """ Uses RequestFactory to construct Requests and returns an iterable of 
#         Requests which will be sent to the LM.

#         :param doc:
#             The document as returned from training_docs, validation_docs, or test_docs.
#         :param ctx: str
#             The context string, generated by fewshot_context. This includes the natural 
#             language description, as well as the few shot examples, and the question
#             part of the document for `doc`. 
#         """
#         continuation = rf.greedy_until(ctx, ['\n##\n'])  # explanations for MR tends to include \n in between.
#         return continuation

 
# class ENEM_MULTIMODAL_2022(ENEM_MULTIMODAL):
#     """We recomend using this task for zero-shot, because _get_train_examples 
#     returns examples from 2022 exam. To run with few-shot, it is neccessary to
#     remove from test set the documents returned in _get_train_examples.
#     """

#     def download(self, data_dir=None, cache_dir=None, download_mode=None):

#         self.dataset = collections.defaultdict(list)
        
#         fname = os.path.join(self.DATASET_PATH, '2022.json')
#         with open(fname) as f:
#             documents = json.load(f)
        
#         def ignore_question(doc):
#             filters = {
#                 'IU': False,
#                 # 'MR': False,  # uncomment to filter out MR
#                 # 'CE': False,  # uncomment to filter out CE
#                 'ML': False,
#             }
#             for k,v in filters.items():
#                 if doc[k] != v:
#                     return True
#             return False

#         documents = list(filter(lambda doc: not ignore_question(doc), documents))
#         self.dataset['test'] = list(map(self._process_doc, documents))

#     def process_results(self, doc, results):
#         results = super().process_results(doc, results)

#         q_id = int(doc['id'].split('_')[-1])
#         area = ['languages', 'human-sciences', 'natural-sciences', 'mathematics'][int(np.ceil(q_id/45))-1]

#         results[area] = results['acc']
#         # results['c_' + area] = 1  # just to count number of questions per area
#         return results

#     def test_docs(self):
#         return self.dataset['test']

#     def higher_is_better(self):
#         return {
#             "acc": True,
#             '2022': True,
#             'languages': True,
#             'human-sciences': True,
#             'natural-sciences': True,
#             'mathematics': True,
#             'c_languages': True,
#             'c_human-sciences': True,
#             'c_natural-sciences': True,
#             'c_mathematics': True,
#         }
    
#     def aggregation(self):
#         return {
#             "acc": mean,
#             '2022': mean,
#             'languages': mean,
#             'human-sciences': mean,
#             'natural-sciences': mean,
#             'mathematics': mean,
#             'c_languages': sum,
#             'c_human-sciences': sum,
#             'c_natural-sciences': sum,
#             'c_mathematics': sum,
#         }
        

# class ENEM_MULTIMODAL_CoT_2022(ENEM_MULTIMODAL_CoT, ENEM_MULTIMODAL_2022):
#     pass