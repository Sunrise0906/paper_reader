# -*- coding: utf-8 -*-
# Editor: Sunrise
# Date: 2023/7/18 13:49


import pickle
import os
from dotenv import load_dotenv

from paper_processor import Paper
from reader_core import PaperReader

load_dotenv()  # load openai api_key, store your apikey in your local .env file

# load paper
# paper_path = './paper/alexnet.pdf'
# paper_path = './paper/BERT.pdf'
# paper_path = './paper/GLOW.pdf'
# paper_path = './paper/SARS-CoV-2 article.pdf'
paper_path = './paper/1.pdf'

paper = Paper(paper_path)

# set PaperReader
reader = PaperReader(api_key=os.getenv("OPENAI_API_KEY"))
summary = reader.summarize(paper)

# general summary
question = 'Summarize this paper to help reader quickly understand it.'
general = reader.inquiry(paper, question)
paper.set_general(general)

# save paper & load
pickle.dump(paper, open('digested_paper.pkl', 'wb'))
# paper = pickle.load(open('digested_paper.pkl', 'rb'))
# paper.show_info()
