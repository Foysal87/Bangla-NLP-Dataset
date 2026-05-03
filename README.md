# 🇧🇩 Bangla NLP Dataset

<div align="center">

![Bangla NLP](https://img.shields.io/badge/Language-Bangla-green.svg)
![Datasets](https://img.shields.io/badge/Datasets-150+-blue.svg)
![License](https://img.shields.io/badge/License-Open%20Source-brightgreen.svg)
![Contribution](https://img.shields.io/badge/Contributions-Welcome-orange.svg)

**A comprehensive, URL-validated collection of Bangla / Bengali NLP datasets, models, tools, and corpora — for researchers, students, and developers.**

বাংলা ভাষার এনএলপি গবেষণা, শিক্ষা এবং প্রায়োগিক কাজের জন্য একটি যাচাইকৃত সম্পদ-সংগ্রহ।

---

🔄 **Our sbnltk dataset is in LFS mode — clone the repository to download data.**

🚀 **All deep-learning-era datasets are linked below; we'll keep adding new releases.**

</div>

## 📑 Table of Contents

- [📖 About](#-about)
- [🎯 sbnltk Dataset List](#-sbnltk-dataset-list-dump--human-evaluated)
- [🤖 Pre-trained Language Models](#-pre-trained-language-models)
- [📚 Latest 2024–2026 Datasets](#-latest-20242026-datasets)
- [📊 Benchmarking and Evaluation](#-benchmarking-and-evaluation)
- [📰 News, Corpora & Pretraining](#-news-corpora--pretraining-data)
- [🔄 Machine Translation & Paraphrase](#-machine-translation--paraphrase)
- [🎤 Speech (ASR / TTS / Emotion)](#-speech-asr--tts--emotion)
- [😊 Sentiment / Emotion / Sarcasm](#-sentiment--emotion--sarcasm)
- [🛡️ Hate Speech / Toxic / Cyberbullying](#-hate-speech--toxic--cyberbullying)
- [🕵️ Fake News & Misinformation](#-fake-news--misinformation)
- [🏷️ NER / POS / Parsing](#-ner--pos--parsing)
- [❓ Question Answering](#-question-answering)
- [📝 Text Summarization](#-text-summarization)
- [🖊️ OCR, Handwriting & Document Layout](#-ocr-handwriting--document-layout)
- [✋ Sign Language & Multimodal Vision](#-sign-language--multimodal-vision)
- [🗺️ Regional Dialects](#-regional-dialects)
- [🧪 NLI / Bias / Misc](#-nli--bias--misc)
- [🔧 NLP Tools & Libraries](#-nlp-tools--libraries)
- [📄 Research Papers](#-research-papers)
- [🇧🇩 Bangladesh Government Resources](#-bangladesh-government-resources)
- [🔗 Curated Aggregator Lists](#-curated-aggregator-lists-start-here)
- [💡 Motivation & Contribution](#-motivation--contribution)

## 📖 About

This repository contains the **sbnltk datasets** used in the Bangla NLP toolkit [sbnltk](https://github.com/Foysal87/sbnltk), and serves as a **comprehensive, URL-validated catalogue** of publicly available Bangla NLP resources contributed by the worldwide Bangla research community.

> **Validation note:** Every link in this document was tested between 2025–2026. Resources that previously appeared here under fabricated GitHub paths (e.g. `github.com/poetry-bangla/corpus`, `github.com/medical-bangla/medical-translation`, etc.) have been removed. If you find a dead link, please open an issue.

## 🎯 sbnltk Dataset List (DUMP & HUMAN Evaluated)

| Dataset | Description | Link |
|---------|-------------|------|
| **Number List** | Bangla number list | [📥 Download](https://drive.google.com/file/d/1r85Fwkx2EP2nCESDWlBRy_KcVc5ou3bJ/view) |
| **Root Word List** | Bangla root word list | [📥 Download](https://drive.google.com/file/d/1sI1s5J4efXzJYd-7Vel2qm-NEZ1Tdzhq/view) |
| **Word List** | Bangla word list (highest → lowest occurrence) | [📥 Download](https://drive.google.com/file/d/1Q7lsl9ifQTowC10f3zTXX6QAggTVcD10/view) |
| **Wiki Dump** | Bangla wiki dump words | [📥 Download](https://drive.google.com/file/d/1PSWX6HlIUBlwhqi-zK7XunwkzK_mZYSI/view) |
| **POS Tag Static** | Bangla POS-tag static dataset (single word) | [📥 Download](https://drive.google.com/file/d/14TImhPiW3uQ7R5HRvL5jPMq2R59qd9eT/view) |
| **NER Static** | Bangla NER static dataset (single word) | [📥 Download](https://drive.google.com/file/d/1eVQ3f7X74lCxbURlEjOdUWgRikSXIRlm/view) |
| **Stop Words** | Bangla stop-word list | [📥 Download](https://drive.google.com/file/d/10hZ0Eu_jLWmY_kkkdutXOZGUvdLLKHgy/view) |
| **Dump POS Tag** | Bangla dump POS-tag | [📥 Download](https://drive.google.com/file/d/1-oMBFmFVmCNRUQrL0hF9uJB6ytdseKy6/view?usp=sharing) |
| **Question Classification** | Bangla dump question classification dataset | [📥 Download](https://drive.google.com/file/d/1sM2Zo8K1U80rBMdhylr0STYA9-M49xS2/view?usp=sharing) |
| **Sentiment Analysis** | Bangla dump sentiment analysis | [📥 Download](https://drive.google.com/drive/u/2/folders/1RayFcSnmCuTNmH2ojF-xB3y7QyzGQkCw) |
| **Translation Dataset** | Google translation dataset | [📥 Download](https://drive.google.com/drive/u/2/folders/1qdgJfu0zkxww7m9DWfr73PayrK4g-XH3) |
| **NER Enhanced** | Existing NER dataset (modified + Date entity) | [📥 Download](https://drive.google.com/file/d/1UouYz1kPKeje1vSWhDh1Ashr_32-BcaG/view?usp=sharing) |
| **News Articles** | News article dataset | [📥 Download](https://drive.google.com/file/d/1fvtabFEHqSCEILhIxDxYSR6Vb80l3BhK/view?usp=sharing) |
| **POS Converted** | POS-tag converted data | [📥 Download](https://drive.google.com/drive/u/2/folders/1Pv294DshkxFAMfmh9jrZBsS61yLUpc15) |
| **POS Human Evaluated** | POS-tag human-evaluated data | [📥 Download](https://drive.google.com/drive/u/2/folders/16Ihf_4H0_cCyFe5gMK-kx2Lw09C39-Cw) |
| **NER Dump (Both)** | Dump NER (active + passive) | [📥 Download](https://drive.google.com/file/d/1IxQc5QJRA5cXsxE8v8prHbBBSmMl1gtw/view?usp=sharing) |
| **NER Dump (Active)** | Dump NER (active only) | [📥 Download](https://drive.google.com/file/d/1AT4FkyqyioLIc6wy8mo7cv_Q2ZTnGS_1/view?usp=sharing) |
| **Extractive Summarization** | Extractive text summarization | [🔗 GitHub](https://github.com/Abid-Mahadi/Bangla-Text-summarization-Dataset) |
| **Abstractive Summarization** | Abstractive summarization (newspaper) | [📥 Drive](https://drive.google.com/file/d/1T1dN2GZPYfkWQWc49BUAm2wIhIfiAjdq/view?usp=sharing) · [📊 Kaggle](https://www.kaggle.com/towhidahmedfoysal/bangla-summarization-datasetprothom-alo) |
| **Text Classification** | News article classification | [📥 Drive](https://drive.google.com/file/d/1T1dN2GZPYfkWQWc49BUAm2wIhIfiAjdq/view?usp=sharing) · [📊 Kaggle](https://www.kaggle.com/towhidahmedfoysal/bangla-summarization-datasetprothom-alo) |
| **Keywords Classification** | Topic-keyword classification | [📥 Drive](https://drive.google.com/file/d/1T1dN2GZPYfkWQWc49BUAm2wIhIfiAjdq/view?usp=sharing) · [📊 Kaggle](https://www.kaggle.com/towhidahmedfoysal/bangla-summarization-datasetprothom-alo) |

---

## 🤖 Pre-trained Language Models

### BERT-style Encoders

| Model | Description | Params | Link |
|-------|-------------|--------|------|
| **BanglaBERT** | ELECTRA discriminator, SOTA Bangla NLU (BUET CSE NLP) | 110M | [🤗 HF](https://huggingface.co/csebuetnlp/banglabert) · [🔗 GitHub](https://github.com/csebuetnlp/banglabert) |
| **BanglaBERT (Small)** | Lightweight variant | 13M | [🤗 HF](https://huggingface.co/csebuetnlp/banglabert_small) |
| **BanglaBERT (Large)** | Large variant, top scores on BLUB | 335M | [🤗 HF](https://huggingface.co/csebuetnlp/banglabert_large) |
| **BanglishBERT** | Bilingual (Bangla + English) | 110M | [🤗 HF](https://huggingface.co/csebuetnlp/banglishbert) |
| **Bangla BERT Base (sagorsarker)** | Popular community BERT | 110M | [🤗 HF](https://huggingface.co/sagorsarker/bangla-bert-base) |
| **mBERT-Bengali-NER** | Multilingual BERT fine-tuned for NER | — | [🤗 HF](https://huggingface.co/sagorsarker/mbert-bengali-ner) |
| **mBERT-Bengali-TyDiQA-QA** | mBERT fine-tuned for QA | — | [🤗 HF](https://huggingface.co/sagorsarker/mbert-bengali-tydiqa-qa) |
| **sahajBERT** | ALBERT-based collaborative training | 18M | [🤗 HF](https://huggingface.co/neuropark/sahajBERT) |
| **MuRIL** | Google multilingual (17 Indian) | 236M | [🤗 HF](https://huggingface.co/google/muril-base-cased) |
| **IndicBERT** | AI4Bharat (12 Indian) | — | [🤗 HF](https://huggingface.co/ai4bharat/indic-bert) |

### Generative / Seq2Seq Models

| Model | Description | Params | Link |
|-------|-------------|--------|------|
| **BanglaT5** | T5-style seq2seq (BUET) | 247M | [🤗 HF](https://huggingface.co/csebuetnlp/banglat5) |
| **BanglaT5 (small)** | Small T5 variant | 60M | [🤗 HF](https://huggingface.co/csebuetnlp/banglat5_small) |
| **BanglaT5 NMT bn↔en** | Translation seq2seq | — | [🤗 bn→en](https://huggingface.co/csebuetnlp/banglat5_nmt_bn_en) · [🤗 en→bn](https://huggingface.co/csebuetnlp/banglat5_nmt_en_bn) |
| **BanglaT5-Paraphrase** | Paraphrase seq2seq | — | [🤗 HF](https://huggingface.co/csebuetnlp/banglat5_banglaparaphrase) |
| **BanglaByT5** | Byte-level T5 | small | [📄 arXiv 2505.17102](https://arxiv.org/abs/2505.17102) |
| **GPT-2 Bengali** | Flax-community GPT-2 | 117M | [🤗 HF](https://huggingface.co/flax-community/gpt2-bengali) |

### Bangla LLMs (2025)

| Model | Description | Params | Link |
|-------|-------------|--------|------|
| **TigerLLM-1B-it** | Bangla instruction-tuned LLM | 1B | [🤗 HF](https://huggingface.co/md-nishat-008/TigerLLM-1B-it) |
| **TigerLLM-9B-it** | Larger variant, beats GPT-3.5 on Bangla | 9B | [🤗 HF](https://huggingface.co/md-nishat-008/TigerLLM-9B-it) |
| **TituLLMs (1B / 3B)** | Family of Bangla LLMs with benchmarks | 1B / 3B | [📄 arXiv 2502.11187](https://arxiv.org/abs/2502.11187) |
| **TigerLLM Paper** | ACL 2025 short paper | — | [📄 arXiv 2503.10995](https://arxiv.org/abs/2503.10995) · [📄 ACL 2025](https://aclanthology.org/2025.acl-short.69.pdf) |
| **BanglaLLaMA-3-8B-BnWiki-Instruct** | Llama-3 fine-tuned on Bn Wiki | 8B | [🤗 HF](https://huggingface.co/BanglaLLM/BanglaLLama-3-8b-BnWiki-Instruct) |
| **Bangla LLaMA (saiful9379)** | LoRA-tuned LLaMA | — | [🔗 GitHub](https://github.com/saiful9379/Bangla_LLAMA) |

### Speech Models

| Model | Description | Performance | Link |
|-------|-------------|-------------|------|
| **Wav2Vec2-Bengali (300M)** | Self-supervised ASR | 17.8 % WER | [🤗 HF](https://huggingface.co/Tahsin-Mayeesha/wav2vec2-bn-300m) |
| **Wav2Vec2-XLSR Bengali** | XLSR fine-tune | — | [🤗 HF](https://huggingface.co/tanmoyio/wav2vec2-large-xlsr-bengali) |
| **BanglaConformer** | Conformer ASR by Bengali.AI | — | [🤗 HF](https://huggingface.co/bengaliAI/BanglaConformer) |
| **BanglaASR** | Whisper fine-tuned for Bengali | 14.73 % WER | [🤗 HF](https://huggingface.co/bangla-speech-processing/BanglaASR) · [🔗 GitHub](https://github.com/hassanaliemon/BanglaASR) |
| **Whisper (multilingual)** | OpenAI base model — Bn supported | various sizes | [🤗 HF](https://huggingface.co/openai/whisper-large-v3) |

### Word & Sentence Embeddings

| Resource | Description | Link |
|----------|-------------|------|
| **Bangla FastText (sagorsarker)** | 20 M-token wiki-trained skipgram + CBOW | [🤗 HF](https://huggingface.co/sagorsarker/bangla-fasttext) |
| **Bangla Word2Vec (sagorsarker)** | 100-d Wikipedia embeddings | [🤗 HF](https://huggingface.co/sagorsarker/bangla_word2vec) |
| **fastText 157-language Bengali** | Facebook 300-d Wiki + CC | [🌐 fastText](https://fasttext.cc/docs/en/crawl-vectors.html) |
| **Spark NLP `bengali_cc_300d`** | Production embedding | [🔗 Spark NLP](https://sparknlp.org/2021/02/10/bengali_cc_300d_bn.html) |
| **BanglaEmbed** | Cross-lingual distilled sentence embeddings | [📄 arXiv 2411.15270](https://arxiv.org/html/2411.15270v1) |

---

## 📚 Latest 2024–2026 Datasets

These are the most relevant new releases — cite the original authors when used.

| Dataset | Task | Size / Notes | Link |
|---------|------|--------------|------|
| **Bangla-Instruct** | Instruction-tuning | 342 K instruction–response pairs | [🤗 HF](https://huggingface.co/datasets/md-nishat-008/Bangla-Instruct) |
| **Bangla-TextBook** | LM pretraining | 9.9 M tokens, 163 NCTB textbooks | [🤗 HF](https://huggingface.co/datasets/md-nishat-008/Bangla-TextBook) |
| **BanglaSTEM** | Technical-domain MT | 5 K Bn-En STEM sentence pairs | [📄 arXiv 2511.03498](https://arxiv.org/abs/2511.03498) |
| **NCTB-QA** | Educational QA | 87 805 QA pairs (grade 1–10) | [📄 arXiv 2603.05462](https://arxiv.org/abs/2603.05462) |
| **BanglaQuAD** | Open-domain QA | 30 808 QA pairs | [📄 arXiv 2410.10229](https://arxiv.org/html/2410.10229) |
| **ANCHOLIK-NER** | Regional-dialect NER | 17 405 sentences, 5 regions | [📄 arXiv 2502.11198](https://arxiv.org/abs/2502.11198) |
| **BanNERD** (NAACL 2025) | NER, 10 classes / 29 domains | 85 K sentences, 991 K tokens | [🔗 GitHub](https://github.com/eblict-gigatech/BanNERD) |
| **ONUBAD** | Dialect→Standard MT | Chittagong / Sylhet / Barisal | [🔗 ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340925000083) |
| **BanglaDial** | Dialect text corpus | 60 729 entries × 11 dialects | [🔗 PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12597015/) |
| **BIDWESH** | Regional hate speech | Multi-region | [📄 arXiv 2507.16183](https://arxiv.org/html/2507.16183v1) |
| **BanglaTLit** | Romanized→Bn back-transliteration | 42.7 K + 245.7 K pretrain | [📄 ACL 2024](https://aclanthology.org/2024.findings-emnlp.859/) |
| **BanglishRev** | E-commerce code-mix reviews | 1.74 M Daraz reviews | [📄 arXiv 2412.13161](https://arxiv.org/html/2412.13161v2) |
| **BengaliSent140** | Hate vs non-hate fusion | 140 K speeches | [📄 arXiv 2601.20129](https://arxiv.org/html/2601.20129v1) · [🔗 IEEE DataPort](https://ieee-dataport.org/documents/bengalisent140-bengali-hate-speech-fusion-dataset) |
| **BLUCK** | LLM cultural-knowledge benchmark | 2 366 MCQs / 23 categories | [📄 arXiv 2505.21092](https://arxiv.org/abs/2505.21092) |
| **BNLI** (refined) | NLI | Curated entail/contra/neutral | [📄 arXiv 2511.08813](https://arxiv.org/html/2511.08813) |
| **MultiBanAbs** | Multi-domain abstractive sum. | Multi-corpus | [📄 arXiv 2511.19317](https://arxiv.org/abs/2511.19317) |
| **MultiBanFakeDetect** | Multimodal fake news | Text + image | [🔗 ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2667096825000291) |
| **BanFakeNews-2.0** | Fake news (2024) | 47 K real + 13 K fake | [📊 Mendeley](https://data.mendeley.com/datasets/kjh887ct4j/1) |
| **BanglaHealth** | Health-domain paraphrase | 200 K sentences | [🔗 ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340925004299) |
| **BanglaCHQ-Summ** | Consumer-health-question summary | 2 350 pairs (BLP-2023) | [🔗 GitHub](https://github.com/alvi-khan/BanglaCHQ-Summ) |
| **Bangla-MedER** | Medical NER | 2 980 texts, 6 entity types | [📊 Mendeley](https://data.mendeley.com/datasets/jt4gywvwtj/2) |
| **BanglaSarc3** | Sarcasm (ternary) | 12 089 FB comments | [🔗 ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340925006778) |
| **VACASPATI** | Bangla literature corpus | 11 M sentences / 115 M words | [📄 arXiv 2307.05083](https://arxiv.org/abs/2307.05083) |
| **MixSarc** | Code-mix sarcasm/humor/offence | Bn-En transliterated | [📄 arXiv 2602.21608](https://arxiv.org/html/2602.21608) |
| **EmoMix-3L** | Code-mix emotion | 1 071 Bn-Hi-En instances | [🔗 GitHub](https://github.com/GoswamiDhiman/EmoMix-3L) |
| **Bangla-ToCo** | Context-aware toxic | 1 004 FB news comments | [🔗 ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340925009989) |
| **BanglaDocAtlas** | Document-layout, 8 classes | annotated complex docs | [🔗 IEEE](https://ieeexplore.ieee.org/document/11196300/) |
| **FoodBD** | BD cuisine images, 67 categories | 3 523 polygon-annotated meals | [🔗 Springer 2025](https://link.springer.com/article/10.1186/s13104-025-07583-8) |
| **DeshiFoodBD** | BD traditional food images | 5 425 images / 19 dishes | [🔗 Springer](https://link.springer.com/chapter/10.1007/978-981-19-2347-0_50) |

---

## 📊 Benchmarking and Evaluation

### BLUB — Bangla Language Understanding Benchmark

The first comprehensive Bangla NLU benchmark, introduced with BanglaBERT (NAACL 2022).

| Task | Dataset | Metric | Best Model | Score |
|------|---------|--------|-----------|-------|
| Sentiment Classification | SentNoB | Macro-F1 | BanglaBERT | 72.89 |
| Natural Language Inference | XNLI-bn / BNLI | Accuracy | BanglaBERT (Large) | 83.41 |
| Named Entity Recognition | MultiCoNER | Micro-F1 | BanglaBERT (Large) | 79.20 |
| Question Answering | SQuAD-bn / TyDiQA | EM / F1 | BanglaBERT (Large) | 76.10 / 81.50 |

📄 BLUB code & leaderboard: [github.com/csebuetnlp/banglabert](https://github.com/csebuetnlp/banglabert)

### BLUCK — Bangla LLM Cultural & Linguistic Benchmark (2025)

2 366 multiple-choice questions across 23 categories covering Bangladesh culture, history, and Bangla linguistics — designed to probe LLM cultural knowledge. [📄 arXiv 2505.21092](https://arxiv.org/abs/2505.21092)

### Recent Benchmark Datasets

| Dataset | Task | Size | Link |
|---------|------|------|------|
| **BanglaBook** | Sentiment | 158 065 reviews | [🔗 GitHub](https://github.com/mohsinulkabir14/BanglaBook) |
| **SentMix-3L / OffMix-3L** | Code-mix sentiment / offence | ~1 K each | [📄 ACL](https://aclanthology.org/2023.socialnlp-1.3.pdf) |
| **MultiCoNER (Bangla)** | Multilingual complex NER | task | [🔗 multiconer.github.io](https://multiconer.github.io/) |

---

## 📰 News, Corpora & Pretraining Data

| Dataset | Size | Link |
|---------|------|------|
| **Bangla2B+** (BanglaBERT pretraining corpus) | 27.5 GB / 110 sites | [🔗 GitHub](https://github.com/csebuetnlp/banglabert) |
| **BanglaLM** (data-mining corpus) | 14 GB | [📄 IEEE](https://ieeexplore.ieee.org/document/9544818/) |
| **BdNC – Bangladesh National Corpus** ✅ | 40 GB / 3 B+ words | [🔗 corpus.bangla.gov.bd](https://corpus.bangla.gov.bd/) |
| **VACASPATI** literary corpus | 11 M sentences | [📄 arXiv 2307.05083](https://arxiv.org/abs/2307.05083) |
| **CC-100 Bangla** | 8.3 GB | [🔗 StatMT](https://data.statmt.org/cc-100/) |
| **OSCAR Bangla** | 12 GB+ | [🔗 OSCAR](https://oscar-corpus.com/) |
| **AI4Bharat IndicCorp** | 9 B tokens incl. Bangla | [🔗 site](https://indicnlp.ai4bharat.org/corpora/) |
| **AI4Bharat IndicNLP corpus + catalog** | meta-resource | [🔗 corpus](https://github.com/AI4Bharat/indicnlp_corpus) · [🔗 catalog](https://github.com/AI4Bharat/indicnlp_catalog) |
| **Bangla Wikipedia Corpus (Kaggle)** | wiki-text | [📊 Kaggle](https://www.kaggle.com/datasets/shazol/bangla-wikipedia-corpus) |
| **Wikipedia bnwiki dumps** | latest dumps | [🔗 dumps.wikimedia.org](https://dumps.wikimedia.org/bnwiki/latest/) |
| **Leipzig Bengali corpora (2021)** | 1.65 M sentences | [🔗 corpora.uni-leipzig.de](https://corpora.uni-leipzig.de/en?corpusId=ben_wikipedia_2021) |
| **Wiki Articles (Kaggle)** | wiki snapshot | [📊 Kaggle](https://www.kaggle.com/abyaadrafid/bnwiki) |
| **40k News Articles** | 40 K | [📊 Kaggle](https://www.kaggle.com/zshujon/40k-bangla-newspaper-article) |
| **Largest Bangla Newspaper** | large multi-paper | [📊 Kaggle](https://www.kaggle.com/ebiswas/bangla-largest-newspaper-dataset) |
| **bdNews24 corpus** | bdnews24 articles | [📊 Kaggle](https://www.kaggle.com/peyash/bdnews24-corpus) |
| **Bangladesh Protidin** | Bangladesh Protidin news | [📊 Kaggle](https://www.kaggle.com/shakirulhasan/bangla-news-datasets-from-bdpratidin) |
| **csebuetnlp/xlsum** | XL-Sum (Bangla subset) | [🤗 HF](https://huggingface.co/datasets/csebuetnlp/xlsum) |
| **csebuetnlp/dailydialogue_bn** | Daily-dialogue translated | [🤗 HF](https://huggingface.co/datasets/csebuetnlp/dailydialogue_bn) |
| **goru001/nlp-for-bengali** | ULMFiT model + Wiki / news data | [🔗 GitHub](https://github.com/goru001/nlp-for-bengali) |
| **masiur/Bangla-Corpus** | Open community corpus | [🔗 GitHub](https://github.com/masiur/Bangla-Corpus) |

---

## 🔄 Machine Translation & Paraphrase

| Dataset | Description | Link |
|---------|-------------|------|
| **csebuetnlp/BanglaNMT** | 2.38 M Bn-En pairs (133 MB) | [🤗 HF](https://huggingface.co/datasets/csebuetnlp/BanglaNMT) · [🔗 GitHub](https://github.com/csebuetnlp/banglanmt) |
| **AI4Bharat Samanantar** | 49.6 M sentence pairs across Indic languages | [🤗 HF](https://huggingface.co/datasets/ai4bharat/samanantar) · [🔗 site](https://indicnlp.ai4bharat.org/samanantar/) |
| **SUPara0.8M** | Balanced En-Bn corpus | [🔗 IEEE DataPort](https://ieee-dataport.org/documents/supara08m-balanced-english-bangla-parallel-corpus) |
| **BanglaSTEM** | 5 K STEM Bn-En pairs | [📄 arXiv 2511.03498](https://arxiv.org/abs/2511.03498) |
| **WMT24 Bangla seed dataset** | High-quality manual translation | [📄 ACL 2024](https://aclanthology.org/2024.wmt-1.42.pdf) |
| **BanglaParaphrase** | 466 K paraphrase pairs (AACL 2022) | [🤗 HF](https://huggingface.co/datasets/csebuetnlp/BanglaParaphrase) · [🔗 GitHub](https://github.com/csebuetnlp/banglaparaphrase) |
| **OPUS Collections** | Multi-source parallel corpora | [🔗 OPUS](https://opus.nlpl.eu/) |
| **Bengali Visual Genome 1.0** | 29 K image-caption multimodal | [🔗 LINDAT](https://lindat.mff.cuni.cz/repository/items/b2f9edcf-e357-4b09-a96b-1a4c3ec6d365) |
| **Google Dakshina** | 12 South-Asian transliteration / parallel | [🔗 GitHub](https://github.com/google-research-datasets/dakshina) |
| **BanglaTLit** | Romanized → Bangla | [📄 ACL 2024](https://aclanthology.org/2024.findings-emnlp.859/) |
| **bntranslit** | Transliteration toolkit | [🔗 GitHub](https://github.com/sagorbrur/bntranslit) |
| **Bengali Dictionary (Minhas Kamal)** | Dictionary | [🔗 GitHub](https://github.com/MinhasKamal/BengaliDictionary) |
| **TED2020 (Bangla)** | TED multilingual | [📥 TSV](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/ted2020.tsv.gz) |

---

## 🎤 Speech (ASR / TTS / Emotion)

### ASR / Recognition

| Dataset | Description / Size | Link |
|---------|--------------------|------|
| **OpenSLR-53** | Large Bengali ASR — 196 K utterances / 14.6 GB (Google) | [🔗 OpenSLR](https://www.openslr.org/53/) |
| **OpenSLR (HF mirror)** | All OpenSLR languages | [🤗 HF](https://huggingface.co/datasets/openslr/openslr) |
| **OpenSLR-104** | Multilingual code-switching | [🔗 OpenSLR](https://www.openslr.org/104/) |
| **Bengali Common Voice (Mozilla)** | 399 h+ / 19 817 contributors (v9.0) | [🔗 Mozilla](https://commonvoice.mozilla.org/bn) |
| **Bengali.AI OOD-Speech** | 1 177 h / 22 645 speakers — largest Bn ASR | [🔗 Bengali.AI](https://bengaliai.github.io/asr) |
| **FLEURS Bangla** | Cross-lingual 12 h | [🤗 HF](https://huggingface.co/datasets/google/fleurs) |
| **BanglaASR Dataset** | Fine-tuned ASR | [🔗 GitHub](https://github.com/hassanaliemon/BanglaASR) |
| **SUST-CSE-Speech / banspeech** | TTS / ASR corpus | [🤗 HF](https://huggingface.co/datasets/SUST-CSE-Speech/banspeech) |
| **SKNahin / open-large-bengali-asr-data** | Large open ASR | [🤗 HF](https://huggingface.co/datasets/SKNahin/open-large-bengali-asr-data) |
| **Bangla-Speech-Corpora (Bangla-Language-Processing)** | Cleaned TTS-ready speech | [🔗 GitHub](https://github.com/Bangla-Language-Processing/Bangla-Speech-Corpora) |

### TTS

| Dataset | Description | Link |
|---------|-------------|------|
| **OpenSLR-37** | High-quality Google Bengali TTS | [🔗 OpenSLR](https://www.openslr.org/37/) |
| **Bengali.AI TTS dataset** | Studio-quality TTS | [🔗 site](https://www.bengali.ai/datasets) |
| **bangla-tts (zabir-nabil)** | Real-time multilingual synthesis | [🔗 GitHub](https://github.com/zabir-nabil/bangla-tts) |

### Speech Emotion

| Dataset | Description | Link |
|---------|-------------|------|
| **SUBESCO** | 7 K utterances / 7 emotions, gender-balanced | [📄 PLOS One](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0250173) |
| **BanglaSER** | 1 467 utterances / 34 speakers / 5 emotions | [🔗 ScienceDirect](https://www.sciencedirect.com/science/article/pii/S235234092200302X) |
| **KBES** | Realistic speech-emotion w/ intensity | [🔗 ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340923008107) |
| **BANSpEmo** | Bangla emotional speech | [📄 arXiv 2312.14020](https://arxiv.org/pdf/2312.14020) |

### Speech-to-Text Toolkits

| Tool | Notes | Link |
|------|-------|------|
| **BanglaSpeech2Text** | Whisper-FT offline ASR (mp3/mp4/wav) | [🔗 GitHub](https://github.com/shhossain/BanglaSpeech2Text) |

---

## 😊 Sentiment / Emotion / Sarcasm

| Dataset | Size / Notes | Link |
|---------|--------------|------|
| **BanglaBook** | 158 K book reviews | [🔗 GitHub](https://github.com/mohsinulkabir14/BanglaBook) |
| **SentNoB** | Noisy social-media sentiment | [📊 Kaggle](https://www.kaggle.com/datasets/cryptexcode/sentnob-sentiment-analysis-in-noisy-bangla-texts) |
| **EmoNoBa** | 22 698 comments / 6 emotions (AACL 2022) | [📊 Kaggle](https://www.kaggle.com/datasets/saifsust/emonoba) · [📄 ACL](https://aclanthology.org/2022.aacl-short.17/) |
| **BanglaEmotion (shaoncsecu)** | Emotion benchmark | [🔗 GitHub](https://github.com/shaoncsecu/BanglaEmotion) |
| **MONOVAB** | Multi-label emotion | [📄 paper](https://scispace.com/pdf/monovab-an-annotated-corpus-for-bangla-multi-label-emotion-tqpkarotca.pdf) |
| **BAN-ABSA / BANGLA-ABSA** | 9 009 aspect-level comments | [📄 arXiv 2012.00288](https://arxiv.org/abs/2012.00288) · [📊 Mendeley](https://data.mendeley.com/datasets/998m4jy3m9/2) |
| **BanglaSenti (lexicon)** | 61 582 polarity words | [🔗 GitHub](https://github.com/fahad35/BanglaSenti-A-Dataset-of-Bangla-Words-for-Sentiment-Analysis) |
| **banglanlp/bangla-sentiment-classification** | Compiled benchmarking sets | [🔗 GitHub](https://github.com/banglanlp/bangla-sentiment-classification) |
| **Ayubur sentiment datasets** | Multiple legacy datasets | [🔗 GitHub](https://github.com/Ayubur/bangla-sentiment-analysis-datasets) |
| **BanglaSarc** | 5 112 sarcasm samples | [📊 Kaggle](https://www.kaggle.com/datasets/sakibapon/banglasarc) |
| **BanglaSarc3** | 12 089 ternary-class sarcasm | [🔗 ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340925006778) |
| **BnSentMix** | 20 K Bn-En code-mix sentiment | [📄 paper](https://www.researchgate.net/publication/383236190_BnSentMix) |
| **SentMix-3L / OffMix-3L** | Bn-En-Hi code-mix | [📄 ACL](https://aclanthology.org/2023.socialnlp-1.3.pdf) |
| **Drama Review** | Bengali drama reviews | [📊 Figshare](https://figshare.com/articles/dataset/Bangla_Bengali_Drama_Review_Dataset/13162085) |
| **YouTube Sentiment / Emotion** | YT comments | [📊 Kaggle](https://www.kaggle.com/nit003/bangla-youtube-sentiment-and-emotion-datasets) |
| **News Comments Sentiment** | Bn news comments | [📊 Kaggle](https://www.kaggle.com/mobassir/bengali-news-comments-sentiment) |
| **News Headline Categories** | Headline classification | [📊 Kaggle](https://www.kaggle.com/kaisermasum24/bengali-news-headline-categories) |
| **Big News Classification** | Large news classifier | [📊 Kaggle](https://www.kaggle.com/hasanmoni/bengali-news-classification) |
| **News Article Classification (IndicNLP)** | Indic news classifier | [📊 Kaggle](https://www.kaggle.com/csoham/classification-bengali-news-articles-indicnlp) |
| **Bengali-Banglish Emotion** | Mixed-script | [📊 Mendeley](https://data.mendeley.com/datasets/4dnrwbxt8n/2) |
| **E-commerce Sentiment + Emotion** | 78 130 Daraz / Pickaboo reviews | [🔗 ScienceDirect](https://www.sciencedirect.com/science/article/pii/S235234092400026X) |
| **BanglishRev** | 1.74 M Daraz code-mix reviews | [📄 arXiv 2412.13161](https://arxiv.org/html/2412.13161v2) |

---

## 🛡️ Hate Speech / Toxic / Cyberbullying

| Dataset | Size / Notes | Link |
|---------|--------------|------|
| **Bengali-Hate-Speech (rezacsedu)** | 6 418 / 5 categories | [🔗 GitHub](https://github.com/rezacsedu/Bengali-Hate-Speech-Dataset) · [📊 UCI](https://archive.ics.uci.edu/dataset/719/bengali+hate+speech+detection+dataset) |
| **Bengali Hate Speech (naurosromim)** | Annotated hate dataset | [📊 Kaggle](https://www.kaggle.com/datasets/naurosromim/bengali-hate-speech-dataset) |
| **BIDWESH** | Regional-based hate speech | [📄 arXiv 2507.16183](https://arxiv.org/html/2507.16183v1) |
| **BengaliSent140** | 140 K hate vs non-hate | [🔗 IEEE DataPort](https://ieee-dataport.org/documents/bengalisent140-bengali-hate-speech-fusion-dataset) |
| **ToxLex_bn** | 1 959 bigrams from 2.2 M FB comments | [📊 Mendeley](https://data.mendeley.com/datasets/9pz8ssmc49/2) |
| **Multi-Labeled Bengali Toxic** | 16 073 / 7 labels | [🔗 GitHub](https://github.com/deepu099cse/Multi-Labeled-Bengali-Toxic-Comments-Classification) |
| **Bangla-ToCo** | 1 004 context-aware toxic | [🔗 ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340925009989) |
| **Bangla Multilabel Cyberbully** | 12 557 (5 classes) | [📊 Mendeley](https://data.mendeley.com/datasets/sz5558wrd4/3) |
| **Bangla Social-Media Cyberbullying** | YT/FB/IG/TikTok | [🔗 IEEE DataPort](https://ieee-dataport.org/documents/bangla-social-media-cyberbullying-dataset-0) |
| **Code-mixed Chaos (Banglish toxic)** | 10 234 multi-label | [📊 Mendeley](https://data.mendeley.com/datasets/23dp3t88vk/1) |
| **BanglaMedia** | 7 725 YT comments — 10 topics, 4 sentiments | [📊 Mendeley](https://data.mendeley.com/datasets/xyxb5kryx3/1) |
| **VITD (BLP-2023 violence)** | Violence-inciting text — 3 classes | [🔗 BLP](https://blp-workshop.github.io/) |

---

## 🕵️ Fake News & Misinformation

| Dataset | Size | Link |
|---------|------|------|
| **BanFakeNews** (LREC 2020) | 50 K news | [📊 Kaggle](https://www.kaggle.com/datasets/cryptexcode/banfakenews) · [📄 arXiv 2004.08789](https://arxiv.org/abs/2004.08789) |
| **BanFakeNews-2.0** (2024) | 47 K real + 13 K fake | [📊 Mendeley](https://data.mendeley.com/datasets/kjh887ct4j/1) |
| **MultiBanFakeDetect** | Multimodal text + image | [🔗 ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2667096825000291) |
| **Rowan1224/FakeNews** | Code & data | [🔗 GitHub](https://github.com/Rowan1224/FakeNews) |
| **DataCOVID19** | 14 571 COVID misinformation | [🔗 Springer](https://link.springer.com/chapter/10.1007/978-3-031-73318-5_11) |

---

## 🏷️ NER / POS / Parsing

| Dataset | Size / Notes | Link |
|---------|--------------|------|
| **B-NER** (IEEE Access) | 22 144 sentences | [🔗 IEEE](https://ieeexplore.ieee.org/document/10103464/) |
| **BanNERD** (NAACL 2025) | 85 K sentences / 991 K tokens / 10 classes / 29 domains | [🔗 GitHub](https://github.com/eblict-gigatech/BanNERD) |
| **NER-Bangla-Dataset (MISabic)** | 70 K sentences / 5 types | [🔗 GitHub](https://github.com/MISabic/NER-Bangla-Dataset) |
| **bnlp-resources NER** | Train/dev/test splits | [🌐 banglanlp.github.io](https://banglanlp.github.io/bnlp-resources/ner/) |
| **ANCHOLIK-NER** | Regional NER, 5 regions | [📄 arXiv 2502.11198](https://arxiv.org/abs/2502.11198) |
| **celloscope_bangla_ner_dataset** | 319 K NER | [🤗 HF](https://huggingface.co/datasets/celloscopeai/celloscope_bangla_ner_dataset) |
| **celloscope_bangla_ner_dataset (small)** | 6.57 K | [🤗 HF](https://huggingface.co/datasets/celloscopeai/bangla_ner_dataset) |
| **Bangla-MedER** | Medical NER 2 980 / 6 types | [📊 Mendeley](https://data.mendeley.com/datasets/jt4gywvwtj/2) |
| **Bangla NER (towhidahmedfoysal)** | 400 K word-level | [📊 Kaggle](https://www.kaggle.com/towhidahmedfoysal/bangla-name-entity-recognition) |
| **POS — 3 K sentences** | abhishekgupta92 | [🔗 GitHub](https://github.com/abhishekgupta92/bangla_pos_tagger/tree/master/data) |
| **POS — 100 K+ words (towhidahmedfoysal)** | Word-level POS | [📊 Kaggle](https://www.kaggle.com/towhidahmedfoysal/bangla-parts-of-speechpos-tag) |
| **UD_Bengali-BRU treebank** | 14 UPOS tags, UD v2.9+ | [🌐 universaldependencies.org](https://universaldependencies.org/treebanks/bn_bru/index.html) |

---

## ❓ Question Answering

| Dataset | Size | Link |
|---------|------|------|
| **csebuetnlp/squad_bn** | 118 K train QA | [🤗 HF](https://huggingface.co/datasets/csebuetnlp/squad_bn) |
| **BanglaRQA** | 14 889 QA / 3 K passages (EMNLP 2022) | [🔗 GitHub](https://github.com/sartajekram419/BanglaRQA) |
| **Bengali QA (Mayeesha)** | SQuAD 2.0–style Bn QA | [📊 Kaggle](https://www.kaggle.com/datasets/mayeesha/bengali-question-answering-dataset) |
| **BanglaQuAD** | 30 808 QA pairs | [📄 arXiv 2410.10229](https://arxiv.org/html/2410.10229) |
| **NCTB-QA** | 87 805 educational QA | [📄 arXiv 2603.05462](https://arxiv.org/abs/2603.05462) |
| **doctor_qa_bangla** | 5.14 K medical QA | [🤗 HF](https://huggingface.co/datasets/shetumohanto/doctor_qa_bangla) |
| **TyDiQA (Bengali subset)** | Cross-lingual QA | [🤗 HF](https://huggingface.co/datasets/google-research-datasets/tydiqa) |

---

## 📝 Text Summarization

| Dataset | Size / Notes | Link |
|---------|--------------|------|
| **csebuetnlp/xlsum** | XL-Sum (Bangla subset, BBC) | [🤗 HF](https://huggingface.co/datasets/csebuetnlp/xlsum) |
| **BanglaCHQ-Summ** | 2 350 health-question summary pairs | [🔗 GitHub](https://github.com/alvi-khan/BanglaCHQ-Summ) |
| **MultiBanAbs** | Multi-domain abstractive | [📄 arXiv 2511.19317](https://arxiv.org/abs/2511.19317) |
| **BNLPC + NCTB** | EACL 2021 unsupervised abstractive | [🔗 GitHub](https://github.com/tafseer-nayeem/BengaliSummarization) |
| **BANSData (Prithwiraj)** | News abstractive | [📊 Kaggle](https://www.kaggle.com/datasets/prithwirajsust/bengali-news-summarization-dataset) |
| **Bengali Text Summarization (Hasan Moni)** | Extractive + abstractive | [📊 Kaggle](https://www.kaggle.com/datasets/hasanmoni/bengali-text-summarization) |
| **BUSUM-BNLP** | Multi-document update summarization | [📊 Kaggle](https://www.kaggle.com/datasets/marwanurtaj/busum-bnlp-dataset-multi-document-bangla-summary) |
| **bnSum_gemma7b-it** | Gemma-7B inst news summary system | [🔗 GitHub](https://github.com/samanjoy2/bnSum_gemma7b-it) |
| **Bangla-Text-summarization-Dataset (Abid)** | Extractive | [🔗 GitHub](https://github.com/Abid-Mahadi/Bangla-Text-summarization-Dataset) |
| **3 Human-Evaluated articles (BNLPC)** | Reference summaries | [🌐 BNLPC](http://www.bnlpc.org) |

---

## 🖊️ OCR, Handwriting & Document Layout

| Dataset | Size / Notes | Link |
|---------|--------------|------|
| **NumtaDB** | 85 K handwritten Bn-digit images | [📊 Kaggle](https://www.kaggle.com/datasets/BengaliAI/numta) |
| **Bengali.AI Handwritten Grapheme Classification** | Kaggle competition | [📊 Kaggle](https://www.kaggle.com/c/bengaliai-cv19) |
| **Ekush** | Handwritten Bn characters (largest) | [🌐 rabby.dev/ekush](https://rabby.dev/ekush/) · [📊 Kaggle](https://www.kaggle.com/datasets/shahariar/ekush) |
| **BN-HTRd** | 788 pages / 150 writers / 108 K words (HTR) | [📊 Mendeley](https://data.mendeley.com/datasets/743k6dm543/1) |
| **BanglaWriting** | Multi-purpose offline HTR | [📄 paper](https://www.researchgate.net/publication/346776090_BanglaWriting) |
| **Bayanno** | Multi-purpose handwriting | [📊 Mendeley](https://data.mendeley.com/datasets/jtpfd6j55n) |
| **Bongabdo** | Bn handwritten script | [📄 arXiv 2101.00204](https://arxiv.org/abs/2101.00204v4) |
| **CMATERdb (1 / 2.1.2)** | 5 K word imgs / 18 K Bn city names | [🔗 site](https://code.google.com/archive/p/cmaterdb/) |
| **BaDLAD** | 33 695 layout samples / 6 domains | [📄 paper](https://www.semanticscholar.org/paper/197b64dc89063dba5383a2b8c4fead7754518299) |
| **BanglaDocAtlas** | 8-class complex Bn document layout | [🔗 IEEE](https://ieeexplore.ieee.org/document/11196300/) |
| **DL Sprint 2.0 (BUET CSE Fest 2023)** | Layout segmentation | [📊 Kaggle](https://www.kaggle.com/competitions/dlsprint2/data) |
| **Bangla License Plate 2.5 K** | 2 519 plate images | [🔗 Zenodo](https://zenodo.org/records/7110401) |
| **BD-ALPDR** | 725 high-res LP images | [🌐 site](https://bdalpdr.github.io/) |
| **Govt. Bangla OCR service** | Free Bangla OCR | [🌐 ocr.bangla.gov.bd](https://ocr.bangla.gov.bd/) |

---

## ✋ Sign Language & Multimodal Vision

| Dataset | Size / Notes | Link |
|---------|--------------|------|
| **BdSLW60** | 60 sign words / 9 307 video trials | [📄 arXiv 2402.08635](https://arxiv.org/abs/2402.08635) |
| **KU-BdSL** | 30 classes / 38 consonants | [📊 Mendeley](https://data.mendeley.com/datasets/scpvm2nbkm/4) |
| **BAUST Lipi** | 18 K imgs / 36 alphabets | [📄 arXiv 2408.10518](https://arxiv.org/html/2408.10518) |
| **BDSL 49** | 29 490 imgs / 49 labels | [📄 arXiv 2208.06827](https://arxiv.org/abs/2208.06827) |
| **BdSL36** | 4 M+ imgs / 36 cats | [📄 paper](https://www.researchgate.net/publication/349567007) |
| **Ego-SLD** | Egocentric Bn sign-language video | [🔗 IEEE DataPort](https://ieee-dataport.org/documents/ego-sld-video-dataset-egocentric-action-recognition-bengali-sign-language-detection) |
| **Bengali Visual Genome 1.0** | Multimodal MT + captioning | [🔗 LINDAT](https://lindat.mff.cuni.cz/repository/items/b2f9edcf-e357-4b09-a96b-1a4c3ec6d365) |
| **BAN-Cap** | English-Bangla image-description | [📄 arXiv 2205.14462](https://arxiv.org/pdf/2205.14462) |
| **BNATURE** | Bn image captioning | [📊 Kaggle](https://www.kaggle.com/datasets/almominfaruk/bnaturebengali-image-captioning-dataset) |
| **BanglaView** | 31 783 imgs / 158 K captions | [📊 Mendeley](https://data.mendeley.com/datasets/rrv8pbxrxv/1) |
| **csebuetnlp/illusionVQA** | Bn-aware VQA | [🤗 HF](https://huggingface.co/datasets/csebuetnlp/illusionVQA-Comprehension) |
| **DeshiFoodBD** | 5 425 BD-cuisine images / 19 dishes | [🔗 Springer](https://link.springer.com/chapter/10.1007/978-981-19-2347-0_50) |
| **FoodBD** | 3 523 polygon-annotated meals (2025) | [🔗 Springer](https://link.springer.com/article/10.1186/s13104-025-07583-8) |
| **BnLiT — Bangla Image-to-Text** | Natural-language image-text | [📊 Kaggle](https://www.kaggle.com/jishan900/bangla-natural-language-image-to-text-bnlit) |

---

## 🗺️ Regional Dialects

| Dataset | Coverage | Link |
|---------|----------|------|
| **ANCHOLIK-NER** | Barishal, Chittagong, Mymensingh, Noakhali, Sylhet | [📄 arXiv 2502.11198](https://arxiv.org/abs/2502.11198) |
| **ONUBAD** | Chittagong / Sylhet / Barisal → Standard Bn | [🔗 ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340925000083) |
| **BanglaDial** | 11 dialects / 60 729 entries | [🔗 PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12597015/) |
| **BIDWESH** | Regional hate speech | [📄 arXiv 2507.16183](https://arxiv.org/html/2507.16183v1) |
| **BanglaCHQ-Summ + dialect benchmarks** | Sylheti / Chittagonian | [📄 ACL 2025](https://aclanthology.org/2025.banglalp-1.18.pdf) |
| **Sylheti → Standard NMT** | Sylheti corpus | [🔗 ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340926001290) |

---

## 🧪 NLI / Bias / Misc

| Dataset | Description | Link |
|---------|-------------|------|
| **csebuetnlp/xnli_bn** | Bangla NLI translated from XNLI | [🤗 HF](https://huggingface.co/datasets/csebuetnlp/xnli_bn) |
| **BNLI (refined)** | Curated entail/contra/neutral | [📄 arXiv 2511.08813](https://arxiv.org/html/2511.08813) |
| **csebuetnlp/BanglaSocialBias** | Social-bias evaluation | [🤗 HF](https://huggingface.co/datasets/csebuetnlp/BanglaSocialBias) |
| **csebuetnlp/BanglaContextualBias** | Contextual-bias evaluation | [🤗 HF](https://huggingface.co/datasets/csebuetnlp/BanglaContextualBias) |
| **csebuetnlp/CrossSum** | Cross-lingual summarization | [🤗 HF](https://huggingface.co/datasets/csebuetnlp/CrossSum) |
| **stopwords-iso/stopwords-bn** | Stopword list | [🔗 GitHub](https://github.com/stopwords-iso/stopwords-bn) |
| **Bangla Plagiarism Dataset** | 59.9 K | [🤗 HF](https://huggingface.co/datasets/zarif98sjs/bangla-plagiarism-dataset) |
| **Banking 14-intents (en + bn + banglish)** | 16.5 K intent samples | [🤗 HF](https://huggingface.co/datasets/learn-abc/banking14-intents-en-bn-banglish) |
| **Massive intent (bn-BD)** | 16.5 K | [🤗 HF](https://huggingface.co/datasets/SetFit/amazon_massive_intent_bn-BD) |
| **BanglaMusicStylo** | 2 824 lyrics / 211 lyricists | [🔗 IEEE DataPort](https://ieee-dataport.org/open-access/banglamusicstylo-stylometric-dataset-bangla-music-lyrics) |
| **Bangla Song Lyrics (genres + artists)** | Bn song-lyric corpus | [📊 Kaggle](https://www.kaggle.com/datasets/meherabhasansajid/bangla-song-lyrics-dataset-with-genres-and-artists) |
| **Bn Numbers w/ Words** | Number-name dataset | [📊 Kaggle](https://www.kaggle.com/jabertuhin/bengali-numbers-with-words) |
| **likhonsheikh/BanglaNLP** | 120 K parallel news pairs | [🤗 HF](https://huggingface.co/datasets/likhonsheikh/BanglaNLP) |

---

## 🔧 NLP Tools & Libraries

### Python Libraries

| Library | Description | Link |
|---------|-------------|------|
| **BNLP (sagorbrur)** | Comprehensive Bengali NLP toolkit | [🔗 GitHub](https://github.com/sagorbrur/bnlp) · `pip install bnlp_toolkit` |
| **BNLTK** | Bangla NLP toolkit (tokenize / stem / POS) | [🔗 GitHub](https://github.com/asraf-patoary/bnltk) |
| **sbnltk** | This repo's toolkit (sentiment / NER / POS / sum) | [🔗 GitHub](https://github.com/Foysal87/sbnltk) |
| **bangla-stemmer** | Lightweight Bn stemmer | [🔗 PyPI](https://pypi.org/project/bangla-stemmer/) |
| **bnunicode** | Bijoy → Unicode normalization | [🔗 GitHub](https://github.com/mnansary/bnunicode) |
| **Indic NLP Library** | Multi-Indic processing / transliteration | [🔗 GitHub](https://github.com/anoopkunchukuttan/indic_nlp_library) |
| **bntranslit** | Bengali transliteration | [🔗 GitHub](https://github.com/sagorbrur/bntranslit) |
| **BanglaSpeech2Text** | Bangla offline ASR | [🔗 GitHub](https://github.com/shhossain/BanglaSpeech2Text) |
| **bangla-tts** | Bangla TTS library | [🔗 GitHub](https://github.com/zabir-nabil/bangla-tts) |
| **BanglaKit organisation** | Tools, datasets, resources | [🔗 GitHub](https://github.com/banglakit) |

### OCR / Vision

| Tool | Notes | Link |
|------|-------|------|
| **EasyOCR** | Built-in Bangla support | [🔗 GitHub](https://github.com/JaidedAI/EasyOCR) |
| **Tesseract OCR** | `ben` traineddata available | [🔗 GitHub](https://github.com/tesseract-ocr/tesseract) |
| **ocr.bangla.gov.bd** | Govt-hosted Bangla OCR service | [🌐 site](https://ocr.bangla.gov.bd/) |

---

## 📄 Research Papers

### Foundational

- **BanglaBERT** (NAACL Findings 2022) — [📄 ACL](https://aclanthology.org/2022.findings-naacl.98/) · [📄 arXiv 2101.00204](https://arxiv.org/abs/2101.00204)
- **BanglaNLG / BanglaT5** (EACL 2023) — [📄 arXiv 2205.11081](https://arxiv.org/abs/2205.11081)
- **BanglaParaphrase** (AACL 2022) — [📄 arXiv 2210.05109](https://arxiv.org/abs/2210.05109)
- **Not Low-Resource Anymore: BanglaNMT** (EMNLP 2020) — [📄 ACL](https://aclanthology.org/2020.emnlp-main.207)
- **MuRIL** (Indic multilingual) — [📄 arXiv 2103.10730](https://arxiv.org/abs/2103.10730)
- **Bangla NLP Comprehensive Analysis** — [📄 arXiv 2105.14875](https://arxiv.org/abs/2105.14875)

### LLMs & Generation (2024–2025)

- **TigerLLM** (ACL 2025) — [📄 arXiv 2503.10995](https://arxiv.org/abs/2503.10995)
- **TituLLMs** — [📄 arXiv 2502.11187](https://arxiv.org/abs/2502.11187)
- **BanglaByT5** — [📄 arXiv 2505.17102](https://arxiv.org/abs/2505.17102)
- **BanglaLlama** — [📄 arXiv 2410.21200](https://arxiv.org/pdf/2410.21200)
- **WMT24 Bangla seed** — [📄 ACL](https://aclanthology.org/2024.wmt-1.42.pdf)

### QA / Reading Comprehension

- **BanglaRQA** (EMNLP 2022 Findings) — [📄 ACL](https://aclanthology.org/2022.findings-emnlp.186/)
- **BanglaQuAD** — [📄 arXiv 2410.10229](https://arxiv.org/html/2410.10229)
- **NCTB-QA** — [📄 arXiv 2603.05462](https://arxiv.org/abs/2603.05462)

### NER

- **B-NER** (IEEE Access) — [🔗 IEEE](https://ieeexplore.ieee.org/document/10103464/)
- **ANCHOLIK-NER** — [📄 arXiv 2502.11198](https://arxiv.org/abs/2502.11198)
- **BanNERD** (NAACL 2025) — [🔗 GitHub](https://github.com/eblict-gigatech/BanNERD)

### Summarization

- **BanglaCHQ-Summ** (BLP 2023) — [📄 ACL](https://aclanthology.org/2023.banglalp-1.10/)
- **Bangla extractive (IEEE)** — [🔗 IEEE](https://ieeexplore.ieee.org/document/9667900)
- **MultiBanAbs** — [📄 arXiv 2511.19317](https://arxiv.org/abs/2511.19317)

### Speech

- **BanglaDialecto** (2024) — [📄 arXiv 2411.10879](https://arxiv.org/abs/2411.10879)
- **Bengali Common Voice ASR** — [📄 arXiv 2206.14053](https://arxiv.org/abs/2206.14053)
- **Bengali ASR system survey** — [📄 arXiv 2209.08119](https://arxiv.org/pdf/2209.08119)
- **SUBESCO** (PLOS One 2021) — [📄 PLOS](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0250173)

### Workshops

- **BLP-2023 @ EMNLP (Singapore)** — [🌐 blp-workshop.github.io](https://blp-workshop.github.io/) · [📚 ACL Anthology](https://aclanthology.org/volumes/2023.banglalp-1/)
- **BLP-2025 @ AACL-IJCNLP (Mumbai)** — [🌐 blp-workshop.github.io](https://blp-workshop.github.io/) · shared tasks: Bangla code generation, sentiment, etc.

---

## 🇧🇩 Bangladesh Government Resources

বাংলাদেশ সরকারের তথ্যপ্রযুক্তি ও বাংলা ভাষাগত সম্পদ:

| Resource | বাংলা নাম | Link |
|----------|-----------|------|
| **Bangladesh National Corpus (BdNC)** | বাংলাদেশ জাতীয় কর্পাস (৪০ GB / ৩B+ শব্দ) | [🌐 corpus.bangla.gov.bd](https://corpus.bangla.gov.bd/) |
| **Govt. Bangla OCR** | সরকারি বাংলা OCR সেবা | [🌐 ocr.bangla.gov.bd](https://ocr.bangla.gov.bd/) |
| **Accessible Dictionary** | প্রতিবন্ধী-বান্ধব বাংলা অভিধান | [🌐 accessibledictionary.gov.bd](https://accessibledictionary.gov.bd/) |
| **EBLICT Project** | তথ্যপ্রযুক্তিতে বাংলা ভাষা সমৃদ্ধকরণ প্রকল্প | [🌐 eblict.portal.gov.bd](https://eblict.portal.gov.bd/) |
| **IPA — Information Processing Authority** | বাংলা ভাষা প্রক্রিয়াকরণ কর্তৃপক্ষ | [🌐 ipa.bangla.gov.bd](https://ipa.bangla.gov.bd/about_us) |
| **Bangladesh Computer Council (BCC)** | বাংলাদেশ কম্পিউটার কাউন্সিল | [🌐 bcc.gov.bd](https://bcc.gov.bd/pages/static-pages/688a355cf1fb5aa36c85a11f) |
| **Bangla Academy** | বাংলা একাডেমি | [🌐 banglaacademy.gov.bd](https://banglaacademy.gov.bd/) |

---

## 🔗 Curated Aggregator Lists (start here)

| Resource | Maintainer | Link |
|----------|------------|------|
| **bnlp-resources** | banglanlp | [🔗 GitHub](https://github.com/banglanlp/bnlp-resources) · [🌐 site](https://banglanlp.github.io/bnlp-resources/) |
| **awesome-bangla** | banglakit | [🔗 GitHub](https://github.com/banglakit/awesome-bangla) |
| **bangla-corpus** | sagorbrur | [🔗 GitHub](https://github.com/sagorbrur/bangla-corpus) |
| **Awesome_Bangla_Datasets** | sabbirhossainujjal | [🔗 GitHub](https://github.com/sabbirhossainujjal/Awesome_Bangla_Datasets) |
| **AI4Bharat Indic NLP catalog** | AI4Bharat | [🔗 GitHub](https://github.com/AI4Bharat/indicnlp_catalog) |
| **Mahadih534 — Bangla NLP HF collection** | Mahadih534 | [🤗 HF](https://huggingface.co/collections/Mahadih534/bangla-nlp-datasets-66d2d574b003aeb6d1be6f96) |
| **Mahadih534 — Bangla TTS collection** | Mahadih534 | [🤗 HF](https://huggingface.co/collections/Mahadih534/bangla-tts-datasets) |
| **Mahadih534 — Bangla LLM finetuning** | Mahadih534 | [🤗 HF](https://huggingface.co/collections/Mahadih534/bangla-datasets-for-llms-finetuning-663f2f6d6304d377fce1925a) |
| **csebuetnlp organization** | BUET CSE NLP | [🤗 HF](https://huggingface.co/csebuetnlp) |
| **Sudipta Kar Bangla NLP resources** | personal | [🌐 site](https://sudiptakar.info/bangla_nlp_resources/) |
| **হাতেকলমে Bangla NLP (Rakibul Hassan)** | educational notebooks | [🔗 GitHub](https://github.com/raqueeb/nlp_bangla) · [🌐 book](https://aiwithr.github.io/nlpbook/) |
| **GitHub topic: bangla-nlp** | community-tagged | [🌐 topic](https://github.com/topics/bangla-nlp) |
| **GitHub topic: bengali-nlp** | community-tagged | [🌐 topic](https://github.com/topics/bengali-nlp) |
| **Awesome Public Datasets (general NLP)** | community | [🔗 GitHub](https://github.com/awesomedata/awesome-public-datasets) |
| **NLP Datasets Collection** | community | [🔗 GitHub](https://github.com/niderhoff/nlp-datasets) |

---

## 💡 Motivation & Contribution

Bangla is the 6th-most-spoken language in the world (~270 million native speakers) but remains classified as low-resource in NLP. This repository exists to **make every Bangla NLP resource one click away** — accurately documented and free of fabricated links.

### How to Get Started

1. **For Pre-trained Models** — visit the HuggingFace links above and load directly with `transformers`.
2. **For Tools** — `pip install bnlp_toolkit` or `pip install bnltk`.
3. **For Datasets** — follow individual links; honor each dataset's license (most are CC BY-NC-SA 4.0).
4. **For Research** — start with the BLP workshop proceedings and the latest 2024–2026 release table.

### Contributing

- 📝 Submit new datasets via pull request — include a working URL and one-line description.
- 🐛 Report broken / fabricated links by opening an issue.
- 🔬 Tag your own paper to share with the community.
- ✅ Every PR is welcome — please verify URLs return 200 before submitting.

---

<div align="center">

**⭐ If this repository helps your work, please give it a star! ⭐**

**🤝 Contributions, corrections, and additions are very welcome.**

**🌟 Thanks to every researcher and developer pushing Bangla NLP forward.**

---

## ☕ Support This Project

If this resource has been helpful, you can support its maintenance:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-☕-yellow.svg?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white)](https://www.buymeacoffee.com/towhid_foysal_123)

</div>
