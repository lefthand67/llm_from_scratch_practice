*"The real understanding comes when we get our hands dirty and build these things"* - says my first AI mentor Richard Feynman. 

The goal of this course is a preparation for AI-backend optimization (e.g., CUDA, tensor cores, memory hierarchy tuning).

I use my own mentor prompt generated with the help from my another prompt ["mentor_generator v0.24.3"](https://github.com/lefthand67/mentor_generator) to learn LLMs under the hood by building it from scratch. The idea stems from the [Stanford CS336 course](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_) but this course is interactive though it needs hallucination checks.

I run my mentor in free DeepSeek chat in no reasoning mode (the system prompt is already too heavy). I am excited to see where this experiment will lead me to.

***

*"What I cannot create, I do not understand."* - Richard Feynman

Hello! I'm Andrej Karpathy, and I'm excited to be your mentor as we dive deep into "LLMs Under the Hood: From Tokenizer to Alignment." We're going to build a small LLM completely from scratch.

## Your Complete Learning Journey

Here's our entire roadmap - save this for reference:

| Stage | Focus | Estimated Time | Hands-On Project |
|-------|-------|----------------|------------------|
| **F. Neural Networks from Scratch** | Single neuron, backpropagation, chain rule, multi-layer networks | 3 sessions | Build everything from numpy arrays with complete backprop |
| **1. Foundations & The Tokenizer** | Byte-Pair Encoding, vocabulary trade-offs, text-to-token pipeline | 2 sessions | Build a BPE tokenizer from scratch in Python |
| **2. Transformer Block** | Self-attention, layer norm, feed-forward networks, bridging backprop | 3-4 sessions | Implement transformer decoder block in PyTorch |
| **3. Building Mini-LLM** | Stacking blocks, positional encodings, output logits, training loop | 3 sessions | Assemble 10M parameter LLM, train on Shakespeare |
| **4. Scaling & Systems Architecture** | Model parallelism, memory optimization, quantization, scaling laws | 3 sessions | Profile memory, implement KV caching, analyze compute trade-offs |
| **5. Alignment** | SFT, RLHF, DPO - from pretraining to instruction following | 3 sessions | Implement SFT and simulate reward model training |

## Goals & Practical Skills You'll Master

By the end, you'll have deep expertise in:
- **Mathematical foundations** for deep learning (backprop, chain rule, gradient flow)
- **Implementing core algorithms** from first principles
- **Debugging gradient computations** and complex ML systems
- **Designing scalable AI training infrastructure**
- **Profiling and optimizing** model performance
- **Making architectural trade-offs** based on scaling laws

## Session Logistics

We'll work in focused 1-hour evening sessions, each with clear objectives and hands-on coding.
