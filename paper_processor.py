# -*- coding: utf-8 -*-
# Editor: Sunrise
# Date: 2023/7/16 17:49


from PyPDF2 import PdfReader
import tiktoken


class Paper:

    def __init__(self, filepath):
        self.path = filepath
        self.pdf = PdfReader(filepath)
        self.meta_data = self.pdf.metadata
        self.catalog = None
        self.text = ''
        for page in self.pdf.pages:
            txt = page.extract_text()
            self.text += txt

        self.paper_parts = None  # store separate paper parts
        self.paper_summaries = [('PaperMeta', str(self.meta_data))]
        self.general_summary = None

    def __len__(self):
        return len(self.text)

    def has_catalog(self):
        return self.catalog is not None

    def set_catalog(self, catalog_list):
        self.catalog = list(set(catalog_list))

    def has_general(self):
        return self.general_summary is not None

    def set_general(self, general_summary):
        self.general_summary = general_summary

    def split_paper(self):
        # split paper parts by catalog
        if self.catalog is None:
            raise RuntimeError('Catalog is empty. Cannot split paper parts!')

        sections = self.catalog
        section_index = []
        text = self.text

        # find starting position of each part
        for section in sections:
            idx = text.find(section)
            if idx != -1:
                section_index.append((idx, section))

        section_index.sort()

        # extract paper parts
        paper_parts = []
        for i, (index, section) in enumerate(section_index):
            start = index
            end = section_index[i + 1][0] if i < len(section_index) - 1 else len(text)

            cur_part = text[start:end].strip()
            paper_parts.append((section, cur_part))

        self.paper_parts = paper_parts

    def compute_section_tokens(self, model='gpt-3.5-turbo'):
        if self.paper_parts is None:
            raise RuntimeError('No paper sections exist. Cannot compute!')

        enc = tiktoken.encoding_for_model(model)
        res = []
        for section in self.paper_parts:
            res.append((section[0], len(enc.encode(section[1]))))

        return res

    def show_info(self):
        # show metadata
        # print(f"Title: {self.meta_data['/Title']}")
        # print(f"Author: {self.meta_data['/Author']}")
        # print(f"Published year: {self.meta_data['/Published']}")
        # print(f"Abstract: {self.meta_data['/Description-Abstract']}\n")

        # show catalog
        if self.has_catalog():
            print("Catalog: ", self.catalog)
        else:
            print("The paper has not been cataloged yet.")

        # show length
        print(f"The length is about {len(self.text)}")

        # show summary
        print("################################################################")
        print("The following are the summary of each section of paper. \n")
        if len(self.paper_summaries) > 1:
            for i in self.paper_summaries[1:]:
                print(f"{i[0]}: {i[1].lstrip('- summary:')}")

        print("################################################################")
        if self.has_general():
            print("Here is the general summary of paper digested by chatgpt: ")
            print(self.general_summary)


