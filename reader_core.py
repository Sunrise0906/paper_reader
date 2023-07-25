# -*- coding: utf-8 -*-
# Editor: Sunrise
# Date: 2023/7/17 15:32


import openai
import tiktoken
import json
# import os
# from dotenv import load_dotenv
import tqdm

from paper_processor import Paper
from utils import create_prompt, detect_sections


# load_dotenv()  # load openai api_key, store your apikey in your local .env file

BASE_POINTS = """
1. Who are the authors?
2. What is the process of the proposed method?
3. What is the performance of the proposed method? Please note down its performance metrics.
4. What are the baseline models and their performances? Please note down these baseline methods.
5. What dataset did this paper use?
"""


class ChatGPTCore:

    def __init__(self, api_key, model='gpt-3.5-turbo', temperature=0.2, context_size=4096):
        openai.api_key = api_key
        # openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.context_size = context_size

    # send message to chatgpt core
    def communicate(self, message, only_content=True):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=message,
            temperature=self.temperature
        )

        if only_content:
            return response["choices"][0]["message"]["content"]
        else:
            return response


class ReaderCore:

    def __init__(self, chatgpt_core, key_points=BASE_POINTS):
        self.chatgpt_core = chatgpt_core
        self.enc = tiktoken.encoding_for_model(chatgpt_core.model)  # tokenizer
        self.key_points = key_points
        self.read_buff = 300
        self.question_message = None

    def paper_section_parser(self, filepath, max_length=30):
        """generate paper catalog parsed by chatgpt"""

        system_prompt = """
            You are a researcher helper bot. Now I will give several texts and you help me to find out which of them are the section titles of a research paper.
            You must return me in this json format:
            {
                "titles": ["section title text", "section title text", ...]
            }
            If the title has a number, the number MUST be retained!!!!
            """

        possible_sections = detect_sections(filepath, max_length)
        # print(f"Possible sections: {possible_sections}")
        # print(possible_sections)
        user_prompt = f"These are the texts: {possible_sections}"
        message = [create_prompt(system_prompt), create_prompt(user_prompt, role='user')]

        catalog = json.loads(self.chatgpt_core.communicate(message))

        return catalog

    def read_paper(self, paper):
        # catalog paper
        if not paper.has_catalog():
            print('Beep....Beep....Beep.... Parsing Catalog \n')
            catalog = self.paper_section_parser(paper.path)
            paper.set_catalog(catalog['titles'])
        paper.split_paper()

        # compute tokens for each section
        token_size = paper.compute_section_tokens()
        print('Tokens length for each section: \n', token_size)

        # start summarization
        print()
        print('Beep....Beep....Beep.... I am reading\n')

        reading_prompt = f"""
        You are a researcher helper bot. You can help the user with research paper reading and summarizing. \n
        Now I am going to send you a paper. You need to read it and summarize it for me part by part. \n
        When you are reading, You need to focus on these key points:{self.key_points},

        And You need to generate a brief but informative summary for this part in one sentence.
        Your return format:
        - summary: '...'
        """

        message = [create_prompt(reading_prompt)]

        summaries = []
        for section, content in tqdm.tqdm(paper.paper_parts):
            # add new messages
            new_message = f'now I send you page part {section}: {content}'
            message.append(create_prompt(new_message, role='user'))
            # keep size of messages
            message = self.keep_message(message, self.chatgpt_core.context_size)
            # TODO: current part too large
            if self.count_token(message) >= self.chatgpt_core.context_size:
                # raise RuntimeError("Current section is too large!")
                print('Current section is too large!')
                message.pop()
                continue
            # send
            response = self.chatgpt_core.communicate(message)
            message.append(create_prompt(response))
            # update summary
            summaries.append((section, response))

        paper.paper_summaries.extend(summaries)
        print('Tada... Finish reading. I have built memories for this paper.')
        return paper

    def keep_message(self, message, max_length):
        """keep message length within maxlength by clearing the front of message"""
        total_len = self.count_token(message)
        if total_len < max_length - self.read_buff:
            return message
        else:
            acc = 0
            drop = [False for i in range(len(message))]
            idx = 1
            for send, resp in zip(message[1::2], message[2::2]):
                len_ = self.count_token([send, resp])
                drop[idx] = True
                drop[idx + 1] = True
                idx += 2
                if total_len - (len_ + acc) >= (max_length - self.read_buff):
                    acc += len_
                else:
                    break
            new_msg = []
            for i, d in zip(message, drop):
                if d:
                    continue
                else:
                    new_msg.append(i)
            return new_msg

    def count_token(self, message):
        count = 0
        for i in message:
            count += len(self.enc.encode(i['content']))
        return count

    def inquiry(self, paper, question):
        question_prompt = f"""
        You are a researcher helper bot. You can answer the user's questions about the paper based on the 
        summaries of the paper.
        This is the summary of the paper:
        {paper.paper_summaries}
        """

        if self.question_message is None:
            self.question_message = [create_prompt(question_prompt)]

        inquiry_prompt = f"Now I send you the question: {question} \n"
        self.question_message.append(create_prompt(inquiry_prompt, role='user'))

        response = self.chatgpt_core.communicate(self.question_message)
        self.question_message.append(create_prompt(response))
        return response


"""
    A united paper reader 
"""


class PaperReader:

    def __init__(self, api_key, key_points=BASE_POINTS):
        openai.api_key = api_key
        self.chatgpt_core = ChatGPTCore(api_key=openai.api_key)
        self.reader = ReaderCore(self.chatgpt_core, key_points)

    def summarize(self, paper):
        paper = self.reader.read_paper(paper)
        return paper.paper_summaries

    def inquiry(self, paper, question):
        return self.reader.inquiry(paper, question)

# ------------------------------------
# test code

# # # test section separation
# C = ChatGPTCore()
# R = ReaderCore(C)
# # R.paper_section_parser('./alexnet.pdf')
# R.paper_section_parser('./BERT.pdf')