"""
University Entrance Exam as a Guiding Test for Artificial Intelligence
https://www.ime.usp.br/~ddm/project/enem/ENEM-GuidingTest.pdf

The ENEM Challenge consists in designing an autonomous system that matches the 
performance of a human students on the exam. The overall goal is to foster and 
evaluate the development of Artificial Intelligence techniques that have good 
performance on complex cognitive tasks, not particularly designed for AI systems. 
In addition, this challenge aims to promote and give more visiblity to the 
development of NLP tools for Brazilian Portuguese.

Homepage: https://www.ime.usp.br/~ddm/project/enem
"""
import collections
from io import BytesIO
import json
import numpy as np
import os
import re
from urllib.request import urlopen
import xml.etree.ElementTree as ET 
from zipfile import ZipFile

from lm_eval import utils
from lm_eval.base import Task, rf
from lm_eval.metrics import mean


_CITATION = """
@InProceedings{ ENEM-Challenge,
    author={Silveira, Igor Cataneo and Mau\'a, Denis Deratani},
    booktitle={Proceedings of the 6th Brazilian Conference on Intelligent Systems},
    series={BRACIS},
    title={University Entrance Exam as a Guiding Test for Artificial Intelligence},
    pages={426--431},
    year={2017}
}
"""


PATTERNS_REPLACES = [
    (r'\s*\n+\s*', r' '),  # changing \n to space
    (r'(\s)\1+', r' '),  # changing \n to space
    (r'^\s+', r''),
]


apply_regex = lambda pattern, replace, text: re.sub(pattern, replace, text)


class ENEM(Task):
    VERSION = 0
    DATASET_PATH = 'data/enem'
    DATASET_NAME = None

    use_just_linguistic_and_humanities = False
    tag = None

    # Note: the stats 'EK_only' and 'TC_only' are valid only for use_just_linguistic_and_humanities=True
    enem_stats = {
        '2009-1':    {'EK_only': 0, 'TC_only': 0, 'total': 45}, #
        '2009-2':    {'EK_only': 0, 'TC_only': 0, 'total': 40}, #
        '2010-1':    {'EK_only': 13, 'TC_only': 16, 'total': 45},
        '2010-2':    {'EK_only': 3, 'TC_only': 25, 'total': 40},
        '2011-1':    {'EK_only': 11, 'TC_only': 12, 'total': 45},
        '2011-2':    {'EK_only': 2, 'TC_only': 21, 'total': 40},
        '2012-1':    {'EK_only': 9, 'TC_only': 21, 'total': 45},
        '2012-2':    {'EK_only': 3, 'TC_only': 23, 'total': 40},
        '2013-1':    {'EK_only': 5, 'TC_only': 19, 'total': 45},
        '2013-2':    {'EK_only': 0, 'TC_only': 23, 'total': 40},
        '2014-1':    {'EK_only': 7, 'TC_only': 13, 'total': 45},
        '2014-2':    {'EK_only': 3, 'TC_only': 22, 'total': 40},
        '2015-1':    {'EK_only': 4, 'TC_only': 22, 'total': 45},
        '2015-2':    {'EK_only': 1, 'TC_only': 23, 'total': 40},
        '2016-1':    {'EK_only': 0, 'TC_only': 0, 'total': 45}, #
        '2016-2':    {'EK_only': 0, 'TC_only': 0, 'total': 40}, #
        '2016_2_-1': {'EK_only': 0, 'TC_only': 0, 'total': 45}, #
        '2016_2_-2': {'EK_only': 0, 'TC_only': 0, 'total': 40}, #
        '2017-1':    {'EK_only': 0, 'TC_only': 0, 'total': 45}, #
        '2017-2':    {'EK_only': 0, 'TC_only': 0, 'total': 40}, #
    }

    def download(self, data_dir=None, cache_dir=None, download_mode=None):

        # download and unpack the dataset
        if not os.path.exists(self.DATASET_PATH):
            os.makedirs(self.DATASET_PATH, exist_ok=True)
            URL = "https://www.ime.usp.br/~ddm/project/enem/ENEMdataset.zip"
            http_response = urlopen(URL)
            zipfile = ZipFile(BytesIO(http_response.read()))
            zipfile.extractall(path=self.DATASET_PATH)

        self.dataset = collections.defaultdict(list)

        for exam in self.enem_stats:
            if not self.use_just_linguistic_and_humanities:
                n_questions = None
            else:
                n_questions = self.enem_stats[exam]['total']

            # get the documents
            fname = os.path.join(self.DATASET_PATH, exam + '.xml')
            documents = self._parse_xml(exam.split('-')[0], fname, first_n=n_questions, tag=self.tag)

            # Train and test split are the same. However, in fewshot_examples()
            # we ensure the the prompt for each test example will be composed 
            # only with examples from other exams.
            self.dataset['train'] += documents

        self.dataset['train'] = list(map(self._process_doc, self.dataset["train"]))

    def _parse_xml(self, exam, path, tag=None, first_n=None, verbose=True):
        tree = ET.parse(path)
        root = tree.getroot()

        filters = {
            'IC': 'No',
            'MR': 'No',
            'CE': 'No',
        }

        if tag is not None:
            assert tag in ['TC', 'EK', 'DS', 'TC_only', 'EK_only', 'DS_only'], (
                "Please choose 'TC', 'EK', 'DS', 'TC_only', 'EK_only' or 'DS_only'")

            if tag == 'TC':
                filters['TC'] = 'Yes'
            if tag == 'EK':
                filters['EK'] = 'Yes'
            if tag == 'DS':
                filters['DS'] = 'Yes'
            elif tag == 'TC_only':
                filters['TC'] = 'Yes'
                filters['EK'] = 'No'
                filters['DS'] = 'No'
            elif tag == 'EK_only':
                filters['TC'] = 'No'
                filters['EK'] = 'Yes'
                filters['DS'] = 'No'
            elif tag == 'DS_only':
                filters['TC'] = 'No'
                filters['EK'] = 'No'
                filters['DS'] = 'Yes'

        def ignore_question(child, filters):
            for k,v in filters.items():
                if child.get(k) != v:
                    return True
            return False

        documents = []

        for idx, child in enumerate(root):

            if first_n is not None and idx == first_n:
                break

            if ignore_question(child, filters):
                continue

            header = child.find('header').text
            statement = child.find('statement').text

            if header is None or statement is None:
                continue

            for pattern, replace in PATTERNS_REPLACES:
                header = apply_regex(pattern, replace, header)
                statement = apply_regex(pattern, replace, statement)
                
            options = []

            answers = child.find('answers')
            for option in answers.iter('option'):
                text = option.text
                for pattern, replace in PATTERNS_REPLACES:
                    if text is not None:
                        text = apply_regex(pattern, replace, text)
                options.append(text)

                if option.get('correct') == 'Yes':
                    correct = option.get('id')

            document = {
                'id': exam + '_' + child.get('id'),  # used to filter out largest prompt candidates
                'exam': exam,  # used to get metrics for each exam, and to filter out prompt candidates
                'context': header,
                'question': statement,
                'options': options,
                'label': correct.lower(),
            }
            assert len(document['options']) == 5, print('The document does not have 5 options')
            documents.append(document)

        return documents

    def _get_train_examples(self):
        header = 'Urgência emocional. Se tudo é para ontem, se a vida engata uma primeira e sai em disparada, se não há mais tempo para paradas estratégicas, caímos fatalmente no vício de querer que os amores sejam igualmente resolvidos num átimo de segundo. Temos pressa para ouvir "eu te amo". Não vemos a hora de que fiquem estabelecidas as regras de convívio: somos namorados, ficantes, casados, amantes? Urgência emocional. Uma cilada. Associamos diversas palavras ao AMOR: paixão, romance, sexo, adrenalina, palpitação. Esquecemos, no entanto, da palavra que viabiliza esse sentimento: "paciência". Amor sem paciência não vinga. Amor não pode ser mastigado e engolido com emergência, com fome desesperada. É uma refeição que pode durar uma vida. MEDEIROS, M. Disponível em: http://porumavidasimples.blogspot.com.br. Acesso em: 20 ago. 2017 (adaptado).'
        statement = 'Nesse texto de opinião, as marcas linguísticas revelam uma situação distensa e de pouca formalidade, o que se evidencia pelo(a) '
        options = [
            'impessoalização ao longo do texto, como em: "se não há mais tempo". ',
            'construção de uma atmosfera de urgência, em palavras como: "pressa". ',
            'repetição de uma determinada estrutura sintática, como em: "Se tudo é para ontem". ',
            'ênfase no emprego da hipérbole, como em: "uma refeição que pode durar uma vida". ',
            'emprego de metáforas, como em: "a vida engata uma primeira e sai em disparada". ',
        ]
        explanation_1 = 'A alternativa A. está ERRADA porque impessoalização não é uma marca de pouca formalidade, inclusive o uso do verbo haver representa uma marca de formalidade. A alternativa B. está ERRADA porque o texto até criou uma atmosfera de urgência, embora tenha sido para criticá-la, e discute exatamente a importância da paciência e não da pressa. A alternativa C. está ERRADA porque a estrutura sintática não é repetida sistematicamente ao longo do texto. A alternativa D. está ERRADA porque, embora o texto possua hipérboles, para afirmar que a figura de linguagem é enfatizada, ela deveria aparecer mais vezes. A alternativa E. está CORRETA porque o texto possui comparações implícitas que se caracterizam como metáforas. Logo o texto emprega metáforas. Resposta:'
        explanation_2 = 'O texto é escrito em uma linguagem leve, ágil, e de pouca formalidade. Além disso, possui figuras de linguagem, como metáforas e hipérboles, que não são excludentes. Em uma análise sequencial das alternativas, daria para afirmar que D. e E. estão corretas. Entretanto, observando em detalhes, nota-se que a expressão "emprego de metáforas" mostra ser mais adequada do que "ênfase no emprego da hipérbole", visto que, para afirmarmos que o uso de hipérboles foi enfatizado, a figura de linguagem deveria ter aparecido mais vezes. Isso torna a alternativa E. mais provável de ser CORRETA. Além disso, impessoalização não deve ser apontada como marca de pouca formalidade. Existe também uma atmosfera de urgência, mas que é criticada no texto que destaca a importância da paciência e não da pressa. Por fim, a estrutura sintática não é repetida sistematicamente ao longo do texto. Resposta:'
        document_1 = {
            'id': 'ENEM_2022_21',  # used to filter out from test set
            'exam': '2022',  # used to get metrics for each exam, and to filter out prompt candidates
            'context': header,
            'question': statement,
            'options': options,
            'label': 'e',
            'explanation': explanation_2,
        }

        header = 'Sempre que a relevância do discurso entra em jogo, a questão torna-se política por definição, pois é o discurso que faz do homem um ser político. E tudo que os homens fazem, sabem ou experimentam só tem sentido na medida em que pode ser discutido. Haverá, talvez, verdades que ficam além da linguagem e que podem ser de grande relevância para o homem no singular, isto é, para o homem que, seja o que for, não é um ser político. Mas homens no plural, isto é, os homens que vivem e se movem e agem neste mundo, só podem experimentar o significado das coisas por poderem falar e ser inteligíveis entre si e consigo mesmos. ARENDT, H. A condição humana. Rio de Janeiro: Forense Universitária, 2004.'
        statement = 'No trecho, a filósofa Hannah Arendt mostra a importância da linguagem no processo de'
        options = [
            'entendimento da cultura.',
            'aumento da criatividade.',
            'percepção da individualidade.',
            'melhoria da técnica.',
            'construção da sociabilidade.',
        ]
        explanation_1 = 'A alternativa A. está ERRADA porque Hannah Arendt não trata do entendimento da cultura, mas da relação social entre as pessoas dessa cultura. A alternativa B. está ERRADA porque Hannah Arendt não fala sobre criatividade, mas sobre a construção de laços entre as pessoas. A alternativa C. está ERRADA porque a linguagem é utilizada no oposto da individualidade, em algo mais coletivo e social. A alternativa D. está ERRADA porque o texto não fala de técnica, mas de laços. A alternativa E. está CORRETA porque a nossa sociabilidade se constrói a partir da linguagem, o que faz de nós seres políticos, no sentido de viver em sociedade, em ambientes coletivos. Resposta:'
        explanation_2 = 'Hannah Arendt defende em sua obra que somos seres políticos, no sentido próprio de vivermos em pólis, em ambiente coletivo e social. E essa sociabilidade só é possível por meio do discurso, da linguagem. Desse modo, podemos concluir que a linguagem se apresenta como uma importante ferramenta para a construção da sociabilidade, e portanto a alternativa E. é a CORRETA. Além disso, não se trata do entendimento da cultura, mas da relação social entre as pessoas dessa cultura. Hannah também não fala sobre aumento de criatividade, tampouco sobre técnica. Por fim, a linguagem é utilizada em algo mais coletivo e social, justamente o oposto da individualidade. Resposta:'
        document_2 = {
            'id': 'ENEM_2022_88',  # used to filter out from test set
            'exam': '2022',  # used to get metrics for each exam, and to filter out prompt candidates
            'context': header,
            'question': statement,
            'options': options,
            'label': 'e',
            'explanation': explanation_2,
        }

        header = 'Um casal planeja construir em sua chácara uma piscina com o formato de um paralelepípedo reto retângulo com capacidade para 90 000 L de água. O casal contratou uma empresa de construções que apresentou cinco projetos com diferentes combinações nas dimensões internas de profundidade, largura e comprimento. A piscina a ser construída terá revestimento interno em suas paredes e fundo com uma mesma cerâmica, e o casal irá escolher o projeto que exija a menor área de revestimento. As dimensões internas de profundidade, largura e comprimento, respectivamente, para cada um dos projetos, são: projeto I: 1,8 m, 2,0 m e 25,0 m; projeto II: 2,0 m, 5,0 m e 9,0 m; projeto III: 1,0 m, 6,0 m e 15,0 m; projeto IV: 1,5 m, 15,0 m e 4,0 m; projeto V: 2,5 m, 3,0 m e 12,0 m.'
        statement = 'O projeto que o casal deverá escolher será o'
        options = [
            'I.',
            'II.',
            'III.',
            'IV.',
            'V.',
        ]
        explanation_1 = 'Devemos calcular a área das quatro faces laterais e a área da base inferior (fundo da piscina) e somar essas áreas para obter a área de revestimento. Logo, calculando a área de revestimento de cada projeto, temos: Projeto I: A = 2 x 25 + 2 x 1,8 x (2 + 25) = 147,2; Projeto II: A = 9 x 5 + 2 x 2 x (9 + 5) = 101; Projeto III: A = 15 x 6 + 2 x 1 x (15 + 6) = 132; Projeto IV: A = 4 x 15 + 2 x 1,5 x (15 + 4) = 117; Projeto V: A = 3 x 12 + 2 x 2,5 x (3 + 12) = 111. Logo, o projeto com menor área de revestimento, é o projeto II, portanto a resposta corrreta é B. Resposta:'
        explanation_2 = 'Devemos calcular a área das quatro faces laterais e a área da base inferior (fundo da piscina) e somar essas áreas para obter a área de revestimento. Logo, calculando a área de revestimento de cada projeto, temos: Projeto I: A = 2 x 25 + 2 x 1,8 x (2 + 25) = 147,2; Projeto II: A = 9 x 5 + 2 x 2 x (9 + 5) = 101; Projeto III: A = 15 x 6 + 2 x 1 x (15 + 6) = 132; Projeto IV: A = 4 x 15 + 2 x 1,5 x (15 + 4) = 117; Projeto V: A = 3 x 12 + 2 x 2,5 x (3 + 12) = 111. Logo, o projeto com menor área de revestimento, é o projeto II, portanto a resposta corrreta é B. Resposta:'
        document_3 = {
            'id': 'ENEM_2022_143',  # used to filter out from test set
            'exam': '2022',  # used to get metrics for each exam, and to filter out prompt candidates
            'context': header,
            'question': statement,
            'options': options,
            'label': 'b',
            'explanation': explanation_2,
        }
        return [document_1, document_2, document_3]

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True
        
    def training_docs(self):
        return list(map(self._process_doc, self._get_train_examples()))

    def test_docs(self):
        return self.dataset["train"]

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
            prompt = "Cabeçalho: " + doc["context"] + "\n"
            prompt += "Enunciado: " + doc["question"] + "\nAlternativas:\n"
            for choice, option in zip(choices, doc["options"]):
                prompt += f"{choice.upper()}. {option}\n"
            prompt += "Resposta:"
            return prompt
        choices = ['a', 'b', 'c', 'd', 'e']
        return {
            "query": format_example(doc, choices),
            "choices": doc["options"],
            "gold": choices.index(doc["label"]),
            "id": doc["id"],
            "exam": doc["exam"],
        }

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + ['A.', 'B.', 'C.', 'D.', 'E.'][doc['gold']].upper()

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of 
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural 
            language description, as well as the few shot examples, and the question
            part of the document for `doc`. 
        """
        continuation = rf.greedy_until(ctx, ['\n'])
        return continuation

    def process_results(self, doc, results):
        gold = ['A.', 'B.', 'C.', 'D.', 'E.'][doc['gold']]
        pred = results[0]

        # regex processing. Useful for zero-shot
        match_1 = re.findall(r'(?:|[Ll]etra |[Aa]lternativa )([ABCDE])\.', pred)
        match_2 = re.findall(r'(?:|[Ll]etra |[Aa]lternativa )([ABCDE])', pred)
        if len(match_1) > 0:
            pred = match_1[-1] + '.'
        elif len(match_2) > 0:
            pred = match_2[-1] + '.'
        else:
            print(f'Regex failed at processing {pred=}')
            print(f'{gold=}, {pred=}, {doc["exam"]=}')

        acc = 1. if pred == gold else 0.

        return {
            "acc": acc,
            doc['exam']: acc,
        }
    
    def higher_is_better(self):
        return {
            "acc": True,
            '2009': True,
            '2010': True,
            '2011': True,
            '2012': True,
            '2013': True,
            '2014': True,
            '2015': True,
            '2016': True,
            '2016_2_': True,
            '2017': True,
        }
    
    def aggregation(self):
        return {
            "acc": mean,
            '2009': mean,
            '2010': mean,
            '2011': mean,
            '2012': mean,
            '2013': mean,
            '2014': mean,
            '2015': mean,
            '2016': mean,
            '2016_2_': mean,
            '2017': mean,
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


class ENEM_CoT(ENEM):

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
            prompt = "Cabeçalho: " + doc["context"] + "\n"
            prompt += "Enunciado: " + doc["question"] + "\nAlternativas:\n"
            for choice, option in zip(choices, doc["options"]):
                prompt += f"{choice.upper()}. {option}\n"
            
            prompt += "Explicação: " + doc.get("explanation", "")
            return prompt.strip()
        choices = ['a', 'b', 'c', 'd', 'e']
        return {
            "query": format_example(doc, choices),
            "choices": doc["options"],
            "gold": choices.index(doc["label"]),
            "id": doc["id"],
            "exam": doc["exam"],
        }
    
    def doc_to_target(self, doc):
        return " " + ['A.', 'B.', 'C.', 'D.', 'E.'][doc['gold']].upper()

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of 
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural 
            language description, as well as the few shot examples, and the question
            part of the document for `doc`. 
        """
        continuation = rf.greedy_until(ctx, ['\n##\n'])  # explanations for MR tends to include \n in between.
        return continuation

 
class ENEM_2022(ENEM):
    """We recomend using this task for zero-shot, because _get_train_examples 
    returns examples from 2022 exam. To run with few-shot, it is neccessary to
    remove from test set the documents returned in _get_train_examples.
    """

    def download(self, data_dir=None, cache_dir=None, download_mode=None):

        self.dataset = collections.defaultdict(list)
        
        fname = os.path.join(self.DATASET_PATH, '2022.json')
        with open(fname) as f:
            documents = json.load(f)
        
        def ignore_question(doc):
            filters = {
                'IU': False,
                # 'MR': False,  # uncomment to filter out MR
                # 'CE': False,  # uncomment to filter out CE
                'ML': False,
            }
            for k,v in filters.items():
                if doc[k] != v:
                    return True
            return False

        documents = list(filter(lambda doc: not ignore_question(doc), documents))
        self.dataset['test'] = list(map(self._process_doc, documents))

    def process_results(self, doc, results):
        results = super().process_results(doc, results)

        q_id = int(doc['id'].split('_')[-1])
        area = ['languages', 'human-sciences', 'natural-sciences', 'mathematics'][int(np.ceil(q_id/45))-1]

        results[area] = results['acc']
        # results['c_' + area] = 1  # just to count number of questions per area
        return results

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
        

class ENEM_CoT_2022(ENEM_CoT, ENEM_2022):
    pass