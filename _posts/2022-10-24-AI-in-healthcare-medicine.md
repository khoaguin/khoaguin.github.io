---
layout: post
title: AI in Health & Medicine
author: khoaguin
date: '2022-10-24'
category: ['paper-summarization', 'medical-ai']
keywords: ppml, paper-summarization
usemathjax: true
thumbnail: /assets/img/posts/ai-in-heath-medicine/banner.jpg
permalink: /blog/ai-in-healthcare-medicine
published: false 
---
# AI in Health & Medicine
Hi, in this post, I summarize the paper [AI in health and medicine](https://www.nature.com/articles/s41591-021-01614-0) written by Pranav Rajpurkar, Emma Chen, Oishi Banerjee & Eric J. Topol. The paper is published at the Nature Medicine journal on January, 2022.

## TL;DR

The paper summarizes findings around the progresses in medical AI that have been made in the past 2 years (2020-2022). These findings are gathered and shared via the weekly medical AI newsletter at [https://doctorpenguin.com](https://doctorpenguin.com/). It highlights promising avenues for new progresses such as novel data sources and human-AI collaboration. It discusses challenges facing the field, including technical limitations, building ethical systems, holding people accountability when AI errors occur, respecting patent privacy, and safeguarding against data preach.

## Why?

Artificial intelligence (AI) has a potential to broadly reshape medicine and improve the experiences of both clinicians and patients.

## What?

### Overview of progress, challenges and opportunities for AI in healthcare

| ![space-1.jpg](/assets/img/posts/ai-in-heath-medicine/1.png) |
|:--:|
| *Figure from the paper*|

### Progresses

- **Deep learning for interpretation of medical images**: In recent years, deep learning has achieved remarkable success in image classification. Medical AI research has consequently blossomed in specialties that rely heavily on the interpretation of images, such as radiology, pathology, gastroenterology and ophthalmology.
- **Medical data beyond images**
    - AI has enabled recent advances in the area of biochemistry, improving understanding of the structure and behavior of biomolecules, e.g. AlphaFold for protein folding, protein analysis
    - AI also has applications in genomics, drug discovery…
    - Medical AI also takes advantage of natural language processing, e.g. BioBERT, ClinicalBERT
    - AI methods can also be used on ECG, medical audio data…

### Opportunities for development of AI algorithms

| ![](/assets/img/posts/ai-in-heath-medicine/2.png){: style="max-width: 97%"} |
|:--:|
| *Figure from the paper*|

### Challenges

| ![space-1.jpg](/assets/img/posts/ai-in-heath-medicine/3.png){: style="max-width: 97%"} |
|:--:|
| *Figure from the paper*|


- **Implementation challenges**
    - *Dataset Limitations (especially for multimodal medical AI)*: Gathering data across departments and institutions is difficult and costly. Devices required to obtain medical data can also be prohibitively expensive. Medical data are also in shortage of labels required for supervised learning. Biases in datasets.
    - *Computation Limitations*: Medical images are very large, and the models needed for medical images are very big
    - *Building model trust:* AI systems need to be reliable (high accuracy), convenient to use and easy to integrate into clinical workflows. Medical AI models also need to be explained (intepretable AI).
- **Accountability**: AI systems need to show that they are robust and can generalize across clinical settings and patient populations and ensure that systems protect patient privacy
    - *Shifts in responsibility*: The proliferation of AI also raises concerns around accountability, as it is currently unclear whether developers, regulators, sellers or healthcare providers should be held accountable if a model makes mistakes even after being thoroughly clinically validated.
- **Fairness (Ethical data use)**: Medical datasets are prone to hacks and identity thefts. One solution is decentralizing data storage and federated learning
        
    | ![space-1.jpg](/assets/img/posts/ai-in-heath-medicine/4.png) |
    |:--:|
    | *Figure from the paper*|

## Other Comments

A comprehensive paper from pioneers in the field (Pranav Rajpurkar et al.) that provides a broad overview of the current medical AI landscape.

## Further Reading

[Deep Medicine: How Artificial Intelligence Can Make Healthcare Human Again](https://www.amazon.com/Deep-Medicine-Artificial-Intelligence-Healthcare/dp/1541644638)
