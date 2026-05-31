# DARWIN

### Abstract
DARWIN is an evolutionary GPT model, utilizing a genetic-algorithm like optimization structure with several independent GPT agents being trained individually unique training code. Each iteration, the GPT models are prompted to modify the training code of one another in an attempt to improve their performance in a mutation-like manner, and the best GPT agents are then benchmarked and selected for the next iteration by genetic algorithm. 
For demonstration purposes and due to budget and time constraints, OpenAI API is used to prompt training code improvements and the nanoGPT framework is used as the training code. DARWIN also utilizes persistent JSON-based memory files to track previous reasoning and changes to code to correlate with improvemenst to model performance. and a bidirectional interface for HITL intervention allowing the model to request upgrades such as additional datasets, training scripts, and restructuring of file hierarchies. In experiments, DARWIN achieved a 1.26 percent improvement in model FLOPS utilization (MFU) and a 2.07 percent improvement to perplexity in 5 iterations of training over baseline configurations, demonstrating promising capabilities as a foundation for scaling evolutionary GPT training.

### Paper
[Link](http://arxiv.org/abs/2602.05848)

> The abstract above describes the published **v1** system (nanoGPT trained from scratch,
> OpenAI API for code mutation). The repository is being rewritten to **v2** — LoRA-finetuning
> a capable coding model in an evolutionary loop, with **OpenAI dropped in favor of Anthropic
> (Claude) + a local model**. The ground-truth design is in
> [`ARCHITECTURE.md`](ARCHITECTURE.md); implementation status is tracked in its §10.3.

## Setup

The v2 controller is cross-platform Python managed with [`uv`](https://docs.astral.sh/uv/).
Training/agent/eval work runs in Linux Docker containers (see [`containers/`](containers/)).

```
# Run the test suite
uv run --python 3.14 --extra dev python -m pytest -q
```

The Claude-backed components (the global-memory pass, §7.4; the Claude mutation backend, §4.5)
read the Anthropic API key from the environment:

```
# add to your shell profile
export ANTHROPIC_API_KEY='...'
```

The salvaged SWE-bench harness lives under [`darwin/bench/swe_bench/`](darwin/bench/swe_bench);
when the benchmark runner is wired up (Phase 3, §6), clone the upstream eval repo into it:

```
git clone https://github.com/princeton-nlp/SWE-bench.git darwin/bench/swe_bench/SWE-bench
cd darwin/bench/swe_bench/SWE-bench && git checkout dc4c087c2b9e4cefebf2e3d201d27e36 && pip install -e .
```

## Running

The v2 evolutionary loop is under construction — see [`ARCHITECTURE.md`](ARCHITECTURE.md) §10
for the phased build plan and §10.3 for what currently exists.
