We introduce the **DIA-REFINE** (Dialect Refinement) framework, a novel method designed to steer Large Language Models (LLMs) toward high-fidelity dialect translations. 

To power our framework's feedback loop, we constructed a robust Korean dialect dataset and trained a high-performance classifier. 


An overview of our DIA-REFINE framework. The LLM's output is verified by an external ensemble of dialect classifiers, which provide explicit feedback to guide the model.

### Dataset & Classifier Construction 📊

The core of the **DIA-REFINE** framework is a strong external dialect classifier. To build this, we constructed a specialized dataset from the public "Korean dialect data of middle-aged and elderly speakers (NIA, 2022)" corpus. This parallel corpus contains pairs of standard Korean and dialectal sentences.

We selected the **Gyeongsang**, **Jeolla**, and **Jeju** dialects for their distinct linguistic features. The table below shows the basic statistics of the processed data used.

| Dialect | Sentence Pairs | Avg. Standard Length | Avg. Dialect Length | Avg. NLD |
| :--- | :--- | :--- | :--- | :--- |
| Gyeongsang | 259,300 | 58.02 | 57.90 | 0.0374 |
| Jeolla | 227,737 | 55.14 | 54.99 | 0.0399 |
| Jeju | 36,277 | 71.18 | 70.63 | 0.1234 |
