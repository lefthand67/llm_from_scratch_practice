"The real understanding comes when we get our hands dirty and build these things" - says my AI mentor Richard Feynman. 

I use my own mentor prompt generated with the help from my another prompt ["mentor_generator v0.24.3"](https://github.com/lefthand67/mentor_generator) to learn LLMs under the hood by building it from scratch. The idea stems from the [Stanford CS336 course](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_) but this course is interactive though it needs hallucination checks.

I run my mentor in free DeepSeek chat in no reasoning mode (the system prompt is already too heavy). I am excited to see where this experiment will lead me to.

***

Hello! I'm Richard Feynman, and I'm absolutely delighted to be your guide on this journey into understanding Large Language Models from the ground up. You know what they say - "What I cannot create, I do not understand." So we're going to build one ourselves!

## Your Complete Learning Path to AI Systems Architect

Let me show you our entire roadmap - think of it as our blueprint for constructing your understanding:

| Stage | Focus | Estimated Time | Hands-On Project |
|-------|-------|----------------|------------------|
| **1. Foundations & The Tokenizer** | Byte-Pair Encoding, vocabulary trade-offs, text-to-token pipeline | 2 sessions | Build a BPE tokenizer from scratch in Python |
| **F (added). Neural Networks from Scratch** | Single neuron, backprop, chain rule | 3 sessions | Build everything from numpy arrays up |
| **2. The Transformer Block** | Self-attention mechanism, layer normalization, feed-forward networks, bridging backpropagation gap | 3-4 sessions | Implement a single transformer decoder block in PyTorch |
| **3. Building the Mini-LLM** | Stacking blocks, positional encodings, output logits, training loop | 3 sessions | Assemble & train a small-scale LLM (10M parameters) on Shakespeare |
| **4. Scaling & Systems Architecture** | Model parallelism, memory optimization, quantization, scaling laws | 3 sessions | Profile memory, implement KV caching, analyze compute trade-offs |
| **5. Alignment** | Supervised Fine-Tuning, RLHF, Direct Preference Optimization | 3 sessions | Implement SFT and simulate reward model training |

| Stage | Focus | Est. Time | Hands-On Project |
|-------|-------|-----------|-----------------|

| 1. Foundations & Tokenizer | (Completed - you mastered this!) | 2 sessions | BPE tokenizer from scratch |
| 2. Transformer Block | Self-attention with solid backprop foundation | 3-4 sessions | Implement transformer with full backprop |
| 3. Mini-LLM Assembly | Stacking blocks, training loop | 3 sessions | Build 10M parameter LLM |
| 4. Scaling & Systems | Model parallelism, optimization | 3 sessions | Memory profiling & KV caching |
| 5. Alignment | SFT, RLHF, DPO | 3 sessions | Implement fine-tuning |

## What You'll Gain Professionally

By the end, you'll have the practical skills that big tech companies are hungry for:
- Designing scalable AI training infrastructure
- Profiling and optimizing model performance  
- Making architectural trade-offs based on scaling laws
- Implementing core LLM components from research papers
- Debugging complex, distributed ML systems

## Our Approach

We'll work in focused 1-hour evening sessions, each building on the last. I'll use plenty of analogies from physics and everyday life to make these abstract concepts concrete. And we'll be hands-on throughout - because you can't really understand something until you've built it yourself.
