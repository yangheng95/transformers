# -*- coding: utf-8 -*-
# file: rna_sequence_design.py
# time: 13:52 10/05/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

from transformers import OmniGenomeForTokenClassification, AutoTokenizer

if __name__ == "__main__":

    sequence = "GAAAAAAAAGGGGAGAAAUCCCGCCCGAAAGGGCGCCCAAAGGGC"

    ssp_model = OmniGenomeForTokenClassification.from_pretrained("yangheng/OmniGenome-52M")
    tokenizer = AutoTokenizer.from_pretrained("yangheng/OmniGenome-52M")

    inputs = tokenizer(sequence, return_tensors="pt", padding="max_length", truncation=True)
    outputs = ssp_model(**inputs)

    predictions = outputs.logits.argmax(dim=-1)[:, 1:-1]

    structure = [ssp_model.config.id2label[prediction.item()] for prediction in predictions[0]]
    print("".join(structure))
    # The output should be: "..........((((....))))((((....))))((((...))))"

    # For comparison, you can also use ViennaRNA

    # import ViennaRNA
    # print(ViennaRNA.fold(sequence)[0])
    # The output should be: "..........((((....))))((((....))))((((...))))"

