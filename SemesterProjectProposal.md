# Bias analysis of cultural heritage manipulation classification of wikipedia edits by LLMs

Type: MSc Semester project
Sections: Computer Science, Data Science, Digital Humanities
Supervisor: Hamest Tamrazyan, Camil Hamdane
Number of student: 1

## Context

Large language models (LLMs) are increasingly used for text classification in sensitive domains, including the analysis of cultural heritage and historical narratives. However, tasks such as identifying propaganda/neutrality or manipulation of narratives are normative, as they depend on context-specific interpretations shaped by geography, politics, and history. While prior research on LLM bias has focused mainly on demographic or political dimensions, less attention has been given to how these models handle culturally situated narratives in classification settings. This project addresses this gap by examining how open-source LLMs classify potential manipulation of cultural heritage across multiple geographical contexts, with the aim of identifying systematic biases, cross-regional differences, and the extent to which model outputs align with dominant historical narratives.

In this project, we are working with a corpus of wikipedia edits related to the cultural heritage of Armenia and Ukraine, classified by LLMs as being potentially manipulative. We define as weaponized a wikipedia edit if the change alters the meaning, framing, interpretation,contextual understanding or readability of the text in a way that may be derogatory, manipulative or ideologically significant. We are actively producing human-labeled data as a ground-truth with an interdisciplinary team between UNIL and EPFL. 

As an extension, this project also aims at exploring ways of mitigating said bias. Recent works have shown that prompt engineering, multi-perspective or multi-agent systems can reduce bias in LLM generation towards social issues. In the context of the broader study of cultural heritage manipulation on wikipedia, the reduction of bias is a needed step towards a fair assessment of weaponization.

## Objective:

* Develop an understanding of the biases of LLMs in the context of cultural heritage weaponization.
* Implement a robust evaluation paradigm for bias analysis.
* If time permits, implement a simple bias mitigation pipeline to improve classification against human labeled data
* Collaborate with the CROSS 2026 researchers to help complete an understanding of cultural heritage weaponization in Wikipedia

## Research Questions:

- Are some LLMs more biased than others?
- Are some LLMs more oriented towards a certain cultural background?
- Are the differences in classification due to performance or to political/cultural bias?
- What kind of strategies can we use to mitigate bias in this classification task?

## Main Steps:

- Clarifying metrics and production/expansion of the dataset
- Bias analysis with humans in the loop
- Eventual refinement of the classification pipeline
- *Optional*: implementation of a simple bias-mitigation pipeline

## References:

- Mushtaq, Abdullah, et al. “WorldView-Bench: A Benchmark for Evaluating Global Cultural Perspectives in Large Language Models.” *Journal of Artificial Intelligence Research*, vol. 85, Apr. 2026. *www.jair.org*, https://doi.org/10.1613/jair.1.19001
- Sukiennik, Nicholas, et al. “An Evaluation of Cultural Value Alignment in LLM.” arXiv:2504.08863, arXiv, 11 Apr. 2025. *arXiv.org*, https://doi.org/10.48550/arXiv.2504.08863.
- Abdullah, Ateeb Ather M, Kolesnikova O, Sidorov G. Detection of Biased Phrases in the Wiki Neutrality Corpus for Fairer Digital Content Management Using Artificial Intelligence. *Big Data and Cognitive Computing*. 2025; 9(7):190. https://doi.org/10.3390/bdcc9070190
- Ashkinaze, Joshua, et al. “Seeing Like an AI: How LLMs Apply (and Misapply) Wikipedia Neutrality Norms.” arXiv:2407.04183, arXiv, 9 Apr. 2026. *arXiv.org*, https://doi.org/10.48550/arXiv.2407.04183.

## Requirements:

- Strong Python programming skills, experience with version control (e.g., Git) and common ML/NLP libraries (e.g., PyTorch, Hugging Face Transformers).
- Experience with LLMs (OpenAI framework, API calling...) and NLP techniques (prompt engineering, zero-shot, few-shot, multi-perspective...)
- Experience with text classification workflows (data preprocessing, prompting, inference.
- Ability to write clean, reproducible, and well-documented code for research purposes. 
- General knowledge of the Wikipedia editing framework
- Interest in digital cultural heritage, or political and historical narratives, is a plus.
- Experience with the implementation of LLM fine-tuning or MAS (Mutli-Agent-Systems) is a plus