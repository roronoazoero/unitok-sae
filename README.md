# UniTok SAE: Sparse Autoencoders for Visual Tokenizers (Personal Sandbox)

[cite_start]This repository is a personal sandbox dedicated to experimenting with **Sparse Autoencoders (SAEs)** applied to **UniTok**[cite: 2, 85]. 

[cite_start]The primary goal of this repository is to explore the internal representations, features, and activation patterns of the UniTok visual tokenizer by implementing and training SAE setups[cite: 2]. This is an experimental fork designed for research, local testing, and hacking around with the model's layers.

## About UniTok
[cite_start]Originally proposed by researchers from HKU, ByteDance, and HUST [cite: 90, 91][cite_start], UniTok is a unified visual tokenizer well-suited for both visual generation and understanding tasks[cite: 85, 93]. [cite_start]It is compatible with autoregressive generative models (like LlamaGen), multimodal understanding models (like LLaVA), and unified MLLMs (such as Chameleon and Liquid)[cite: 94].

---

## Usage

### Requirements
* [cite_start]$Python \ge 3.10$ [cite: 133]
* [cite_start]$PyTorch \ge 2.3.1$ [cite: 134]

### Installation
[cite_start]To clone this experimental repository and install the necessary dependencies, run: [cite: 135, 136]

```bash
# Clone this personal fork
git clone [https://github.com/roronoazoero/unitok-sae.git](https://github.com/roronoazoero/unitok-sae.git)
cd unitok-sae

# Install required packages
pip install -r requirements.txt