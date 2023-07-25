# -*- coding: utf-8 -*-
# Editor: Sunrise
# Date: 2023/7/19 15:28


import pickle
import os
from dotenv import load_dotenv

from paper_processor import Paper
from reader_core import PaperReader

load_dotenv()  # load openai api_key, store your apikey in your local .env file

# set PaperReader
reader = PaperReader(api_key=os.getenv("OPENAI_API_KEY"))

# load paper
paper = pickle.load(open('digested_paper.pkl', 'rb'))
# print(paper.meta_data)
paper.show_info()

print(paper.paper_parts[0])
print()
print("#################################################")
print("You can ask me questions about this paper: ")
#
# while True:
#     question = input("\nYour question (input 'q' to quit): ")
#     if question.strip().lower() == 'q':
#         break
#     print(reader.inquiry(paper, question))