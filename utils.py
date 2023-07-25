# -*- coding: utf-8 -*-
# Editor: Sunrise
# Date: 2023/7/18 21:45


from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


def create_prompt(text, role='system'):
    # generate system or role prompt
    assert role in ('system', 'user')

    prompt = {"role": role, "content": text}

    return prompt


def detect_sections(filepath, max_length=30):
    # detect possible sections in a pdf
    # only accepts obj with string length within [5, 30]
    sections = []
    for page_layout in extract_pages(filepath):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text()
                if max_length >= len(text) > 4:
                    sections.append(text)

    return sections


