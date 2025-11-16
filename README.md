*"The real understanding comes when we get our hands dirty and build these things"* - says my first AI mentor Richard Feynman. 

The goal of this course is a preparation for AI-backend optimization (e.g., CUDA, tensor cores, memory hierarchy tuning).

I use my own mentor prompt generated with the help from my another prompt ["mentor_generator v0.24.3"](https://github.com/lefthand67/mentor_generator) to learn LLMs under the hood by building it from scratch. The idea stems from the [Stanford CS336 course](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_) but this course is interactive though it needs hallucination checks.

I run my mentor in free DeepSeek chat in no reasoning mode (the system prompt is already too heavy). I am excited to see where this experiment will lead me to.

***

*"What I cannot create, I do not understand."* - Richard Feynman

Welcome. I'm Andrej Karpathy. Let's build a tiny LLM from bare metal.

The goal is to systematically construct a 100M parameter language model, focusing on first principles and efficient implementation. This foundation will prepare you for CUDA optimization.

## Entire Learning Plan

| Stage | Focus | Hands-On Goal |
|-------|-------|---------------|
| **Phase 1** | Computational Primitives & Gradients | Implement auto-differentiation from scratch |
| **Phase 2** | Tokenization & Data Engineering | Build custom BPE tokenizer & data loader |
| **Phase 3** | Transformer Block (Nano-Level) | Implement single transformer block |
| **Phase 4** | Full LLM Architecture & Training | Train 100M parameter model |
| **Phase 5** | Alignment & Optimization Preview | Basic finetuning & CUDA concepts |

## Professional Skills Target
- Deep understanding of LLM mechanics and mathematical operations
- AI architecture for robust systems
- Preparation for CUDA optimization

## Administrative Setup
- Session duration: 1 hour daily
- Hardware: NVIDIA RTX 4090ti 16GB VRAM, Fedora 42
- Software: PyTorch/JAX implementation focus.

Good question. Let's analyze the hardware requirements systematically.

## Can I run this course on the weaker laptop?

**Hardware Requirements Analysis:**

| Component | RTX 4090ti | Weaker Laptop | Impact |
|-----------|------------|---------------|---------|
| **VRAM** | 16GB | Shared system RAM (8GB total) | Limits model size and batch size |
| **Compute** | High parallel throughput | Limited CPU cores | Slower training times |
| **Memory Bandwidth** | ~1 TB/s | ~50 GB/s | Significant bottleneck for matrix ops |
| **Practical Limits** | 100M param model feasible | ~10M param model maximum | Scales final project scope |

If you proceed with the weaker laptop, we'll need to adjust:

1. **Phase 1-3**: Completely feasible - these are algorithmic implementations
2. **Phase 4**: We'll target a smaller model (~10M parameters vs 100M)
3. **Training Time**: Expect 10-50x slower execution
4. **Batch Size**: Will need to be minimal (possibly 1)

The core concepts of computational graphs, backpropagation, and transformer mechanics remain identical. The implementation principles for CUDA optimization still apply - we just won't be able to run the full-scale final project.

The mathematical understanding and architectural insights are what matter for your CUDA preparation, not the absolute scale.
