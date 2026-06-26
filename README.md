# Enhancing Literature Review with LLM and NLP Methods: Algorithmic Trading Case

## Overview

This repository contains the implementation of the research paper **"Enhancing literature review with LLM and NLP methods. Algorithmic trading case"** by Stanisław Łaniewski and Robert Ślepaczuk. The study demonstrates the application of advanced Natural Language Processing (NLP) techniques, including Large Language Models (LLMs), to analyze and organize knowledge in the domain of algorithmic trading. By leveraging state-of-the-art machine learning methods, the paper outlines how these techniques facilitate the extraction of insights, identification of trends, and comparison of methodologies from a large corpus of academic literature.

The Python/PyTorch code provided in this repository replicates the key methodologies described in the paper, enabling users to understand, experiment, and extend the techniques presented.

---

## Core Concepts

### Problem Statement
Algorithmic trading research is growing at a rapid pace, with diverse methodologies and asset classes being explored. However, synthesizing knowledge from a vast collection of research papers is challenging due to the sheer volume and complexity. This paper addresses the challenge by:
1. Utilizing NLP techniques to filter and analyze relevant papers.
2. Comparing traditional keyword-based approaches with modern topic modeling methods.
3. Demonstrating the utility of LLMs for reasoning, dataset refinement, and answering complex domain-specific questions.

### Key Contributions
1. **Dataset Creation**: Filtering and identifying 14,342 relevant papers from a dataset of 136 million research articles.
2. **Topic Modeling**: Applying dimensionality reduction and clustering techniques to extract themes and trends in algorithmic trading research.
3. **LLM Applications**: Leveraging tools like ChatGPT to refine datasets, decompose complex tasks, and analyze methodologies.
4. **Insights**: Highlighting the growth trajectory of research in algorithmic trading, especially for emerging asset classes like cryptocurrencies.

---

## Code Implementation

### Overview of the Code
The provided Python/PyTorch implementation focuses on replicating the core methodologies described in the paper. The workflow is divided into multiple stages:

1. **Dataset Filtering**:
   - The code includes functions to preprocess and filter a large dataset of research papers based on relevance to algorithmic trading.
   - Traditional keyword-based filtering and embedding-based techniques are implemented.

2. **Topic Modeling and Dimensionality Reduction**:
   - State-of-the-art algorithms like Latent Dirichlet Allocation (LDA) and clustering methods (e.g., k-means) are used to discover key themes in the filtered dataset.
   - Dimensionality reduction techniques such as Principal Component Analysis (PCA) and t-SNE are applied to visualize trends in algorithmic trading research.

3. **LLM Integration**:
   - Large Language Models, such as OpenAI's GPT-based tools, are employed for reasoning tasks.
   - Examples include refining the filtered dataset, deriving insights on asset class trends, and comparing the efficacy of machine learning models.

4. **Case Analysis**:
   - Specific case studies from the filtered dataset are analyzed to demonstrate the evolution of research methodologies and themes.

---

## Installation

### Prerequisites
- Python >= 3.8
- PyTorch >= 1.10
- Additional Python libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `transformers` (for LLM integration)
  - `nltk`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/username/algorithmic-trading-nlp.git
   cd algorithmic-trading-nlp
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Preprocessing and Filtering
Run the preprocessing script to filter the dataset:
```bash
python preprocess.py --input data/research_papers.csv --output data/filtered_papers.csv
```

### Topic Modeling
Execute the topic modeling pipeline:
```bash
python topic_modeling.py --input data/filtered_papers.csv --output results/topics.json
```

### LLM Integration
Analyze filtered papers using an LLM for reasoning tasks:
```bash
python llm_analysis.py --input data/filtered_papers.csv --output results/llm_insights.json
```

### Visualization
Generate visualizations for trends and themes:
```bash
python visualize.py --input results/topics.json --output figures/
```

---

## Results and Insights

### Key Findings
1. **Growth of Algorithmic Trading Research**:
   - The number of publications on algorithmic trading has accelerated faster than overall research output.
   - Cryptocurrencies and machine learning models exhibit the strongest growth trends.

2. **Effectiveness of NLP Techniques**:
   - Modern topic modeling methods, combined with LLMs, significantly outperform traditional keyword-based approaches in extracting meaningful insights.

3. **LLM Utility**:
   - LLMs demonstrated the ability to answer intricate questions, refine datasets, and analyze methodologies efficiently.

---

## Repository Structure

```
.
├── data/                     # Sample datasets
├── results/                  # Output files (topics, insights, etc.)
├── figures/                  # Generated visualizations
├── preprocess.py             # Script for data preprocessing and filtering
├── topic_modeling.py         # Script for topic modeling and clustering
├── llm_analysis.py           # Script for LLM-based reasoning tasks
├── visualize.py              # Script for generating visualizations
├── requirements.txt          # Required Python libraries
└── README.md                 # Project documentation
```

---

## Future Work

1. **Extended LLM Usage**:
   - Explore fine-tuning domain-specific LLMs for better accuracy in literature analysis.

2. **Interactive Tools**:
   - Develop interactive dashboards for exploring trends and insights.

3. **Expanded Dataset**:
   - Apply the methodology to newer datasets that include publications post-2020.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

The implementation is based on methodologies described in the paper **"Enhancing literature review with LLM and NLP methods. Algorithmic trading case"**. We thank the authors for their contributions to the field of NLP and algorithmic trading.

---

## Contact

For questions, issues, or contributions, please contact [Your Name] at [your-email@example.com].