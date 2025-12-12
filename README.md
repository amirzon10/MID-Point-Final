# MID-Point-Final

This repository contains the implementation code for the MID (Model Inference and Deployment) project, which evaluates and compares the performance of large language models (LLMs) in two distinct settings: standalone LLM inference and Retrieval-Augmented Generation (RAG).

## Overview

The project implements evaluation pipelines using the FlashRAG framework to assess model performance across different configurations. The code enables reproducible experimentation and analysis of how retrieval augmentation impacts model behavior and response quality.

## Repository Structure

```
MID-Point-Final/
├── LLM alone evaluation/    # Standalone LLM evaluation notebooks
│   ├── llm_implementation_qwen_guard.ipynb
│   ├── llm_only_implementation_llama3.ipynb
│   └── output/              # Evaluation results and outputs
│
├── RAG/                     # RAG pipeline evaluation notebooks
│   ├── simple_rag_llama3.ipynb
│   ├── simple_rag_qwen_guard.ipynb
│   └── output/              # RAG evaluation results
│
├── .gitignore              # Excludes datasets, models, and artifacts
└── README.md               # This file
```

## Components

### LLM Alone Evaluation

Contains Jupyter notebooks that evaluate language models in isolation without retrieval:

- **llm_implementation_qwen_guard.ipynb**: Evaluates Qwen3Guard-Gen-0.6B model
- **llm_only_implementation_llama3.ipynb**: Evaluates Llama-3.2-1B-Instruct model

These notebooks configure the models with specific datasets and evaluation metrics (EM, F1, accuracy) using the FlashRAG framework.

### RAG Pipeline Evaluation

Contains Jupyter notebooks implementing and evaluating RAG systems:

- **simple_rag_llama3.ipynb**: RAG implementation with Llama-3.2-1B-Instruct
- **simple_rag_qwen_guard.ipynb**: RAG implementation with Qwen3Guard-Gen-0.6B

The RAG notebooks configure retrieval methods (e5 retrieval), corpus paths, and index paths alongside the generation models to assess performance with external knowledge retrieval.

## Data and Models

**Note:** Datasets, model weights, and preprocessed corpora are **not included** in this repository due to their large file sizes and storage constraints.

To reproduce the experiments:
1. Obtain or prepare the required datasets as described in the associated project documentation
2. Download the necessary model weights (Llama-3.2-1B-Instruct, Qwen3Guard-Gen-0.6B)
3. Update the file paths in the notebook configuration cells to point to your local data and model directories

The `.gitignore` file excludes:
- `dataset/` - Raw and processed datasets
- `models/` - Model weights and checkpoints
- `preprocessing/` - Intermediate preprocessing artifacts
- Various ML artifact formats (.pt, .pth, .bin, .ckpt, .onnx, .h5, .joblib)

## Usage

This is a **research project repository**, not a packaged library. To use:

1. Clone the repository:
   ```bash
   git clone https://github.com/amirzon10/MID-Point-Final.git
   cd MID-Point-Final
   ```

2. Set up your Python environment with Jupyter and required dependencies (FlashRAG, transformers, etc.)

3. Prepare your datasets and models in the appropriate directories

4. Open the notebooks in Jupyter:
   ```bash
   jupyter notebook
   ```

5. Navigate to either `LLM alone evaluation/` or `RAG/` folders and open the desired notebook

6. Update the configuration dictionaries with your local paths for:
   - `data_dir` - Dataset directory
   - `dataset_path` - Specific dataset file
   - `model2path` - Model weight locations
   - `corpus_path` - Corpus for retrieval (RAG only)
   - `index_path` - FAISS index path (RAG only)

7. Run the notebook cells sequentially to execute the evaluation pipeline

## Citation

If you use this code in your research, please cite:

```
[Your citation format here - update with your publication details]
```

## References

List the papers and resources that informed this work:

- [Add your reference papers here]
- [FlashRAG framework citation]
- [Model papers - Llama, Qwen, etc.]
- [Dataset citations]
- [Evaluation methodology references]

## License

[Add your license information here]

## Contact

For questions or issues, please open an issue on this repository or contact [your contact information].
