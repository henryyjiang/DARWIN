# DARWIN

### Abstract
DARWIN is an evolutionary GPT model, utilizing a genetic-algorithm like optimization structure with several independent GPT agents being trained individually unique training code. Each iteration, the GPT models are prompted to modify the training code of one another in an attempt to improve their performance in a mutation-like manner, and the best GPT agents are then benchmarked and selected for the next iteration by genetic algorithm. 
For demonstration purposes and due to budget and time constraints, OpenAI API is used to prompt training code improvements and the nanoGPT framework is used as the training code. DARWIN also utilizes persistent JSON-based memory files to track previous reasoning and changes to code to correlate with improvemenst to model performance. and a bidirectional interface for HITL intervention allowing the model to request upgrades such as additional datasets, training scripts, and restructuring of file hierarchies. In experiments, DARWIN achieved a 1.26 percent improvement in model FLOPS utilization (MFU) and a 2.07 percent improvement to perplexity in 5 iterations of training over baseline configurations, demonstrating promising capabilities as a foundation for scaling evolutionary GPT training.

#### Additional implemented features:
- Parallelization
- Memory storage of past iteration modifications
- Agentic troubleshooting of errors when models fail
- model_requests.txt for receiving file structure change, library imports, dataset changes, and additional scripts requests from the model during training
- suggestions_for_model.txt informs models on manual changes and allows users to offer suggestions for direction to take in improving code.

#### To be implemented: 
- SWE-bench performance metrics

### Conclusions
In practice, DARWIN would certainly face time and computation bottlenecks if scaled to large models on the scale of GPT-2 and above. However, the agentic nature of the training makes it uniquely suited to volunteer or grid computing, dividing calculation of local subgradients between volunteers and averaging them for the models trained in each generation. If applied to a product it could be offered as a chatbot service as an executable, using a small portion of GPU or CPU compute power to calculate gradients for training.

## Setup
```
# API keys, add to ~/.bashrc
export OPENAI_API_KEY='...
```

```
# Verify that Docker is properly configured in your environment.
docker build -t my-app -f DOCKERFILE .
```

```
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```
# Clone SWE-bench
cd swe_bench
git clone https://github.com/princeton-nlp/SWE-bench.git
cd SWE-bench
git checkout dc4c087c2b9e4cefebf2e3d201d27e36
pip install -e .
cd ../../
```

## Running Main Controller:
```
python main/main_controller.py
```

## Acknowledgements:
Credit goes to Andrej Karpathy for the nanoGPT framework. Additional inspiration was taken from works Darwin Gödel Machine, Jenny Zhang et. al., and EvoAgentX by Wang et. al.


