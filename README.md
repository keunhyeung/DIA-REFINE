We introduce the **DIA-REFINE** (Dialect Refinement) framework, a novel method designed to steer Large Language Models (LLMs) toward high-fidelity dialect translations. 

<img width="1504" height="761" alt="figure (1)" src="https://github.com/user-attachments/assets/e98dd212-1160-4178-b6e1-ba822b69c648" />

An overview of our DIA-REFINE framework. The LLM's output is verified by an external ensemble of dialect classifiers, which provide explicit feedback to guide the model.

### Dataset Construction 📊

we constructed a specialized dataset from the public ["Korean dialect data of middle-aged and elderly speakers (NIA, 2022)"](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=&topMenu=&srchOptnCnd=OPTNCND001&searchKeyword=%EC%A4%91%EB%85%B8%EB%85%84%EC%B8%B5&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71517) corpus. This parallel corpus contains pairs of standard Korean and dialectal sentences.

We selected the **Gyeongsang**, **Jeolla**, and **Jeju** dialects for their distinct linguistic features. The table below shows the basic statistics of the processed data used.

| Dialect | Sentence Pairs | Avg. Standard Length | Avg. Dialect Length | Avg. NLD |
| :--- | :--- | :--- | :--- | :--- |
| Gyeongsang | 259,300 | 58.02 | 57.90 | 0.0374 |
| Jeolla | 227,737 | 55.14 | 54.99 | 0.0399 |
| Jeju | 36,277 | 71.18 | 70.63 | 0.1234 |

From this large corpus, we created three distinct, non-overlapping sets for training, evaluation, and in-context learning to ensure experimental integrity.

### Classifier Training & Evaluation Set
To build our dialect classifier, we curated dedicated datasets for the Jeolla, Gyeongsang, and Jeju dialects. The data for each dialect was then split into a 9:1 ratio, providing 9,000 samples for training and 1,000 for evaluation.

### Dialect Translation Test Set
For final performance evaluation, we randomly sampled 300 sentence pairs for each of the three target dialects (Jeolla, Gyeongsang, and Jeju). To ensure the sentences had sufficient content for translation, we only included pairs where the sentence length was greater than 30 characters.

### In-Context Learning (ICL) Example Pool
To support our ICL experiments, we created a large example pool containing 10,000 sentence pairs for each dialect. This pool was kept entirely separate from the test set to prevent data leakage and ensure a fair evaluation.
---
### Evaluation Metrics 📈

We found that traditional n-gram-based metrics like BLEU and chrF++ are often unreliable for dialect translation. They tend to reward outputs that simply copy the standard Korean source text rather than generating authentic dialectal features. 

To overcome these limitations and accurately measure the performance of DIA-REFINE, we introduce two novel evaluation metrics.

### 1. Dialect Fidelity Score (DFS)

We designed the Dialect Fidelity Score (DFS) to measure if a translation is linguistically closer to the dialect reference than the standard source. It's calculated from the log ratio of cosine similarities between sentence embeddings. A positive score indicates a successful dialect shift, while a negative score signals a failure.

### 2. Target Dialect Ratio (TDR)

We proposed the Target Dialect Ratio (TDR) to directly measure the model's success rate in generating the correct dialect. It is the percentage of outputs our trained classifier correctly identifies as the target dialect. A higher TDR signifies more consistent dialect generation.
