# Agentic AI Safety Investigation via Prompt Infection

MSc Artificial Intelligence dissertation, University of Surrey.

Multi-agent systems built on large language models are now a common way to handle multi-step tasks, but their security is not well understood. One open question is what happens to a prompt injection attack when the agents can talk to each other, not only to tools or a user. This project studies that question by replicating and extending the Prompt Infection attack framework from Lee and Tiwari (2024).

Prompt infection is an attack where a single malicious prompt copies itself from one agent to the next as they pass messages along a pipeline. Each infected agent re-injects the attack into its own output, so the instruction survives all the way to the final agent, which then carries out the malicious goal. The original paper showed this on OpenAI's GPT models. This study asks whether the same behaviour holds on other model families, and how well the paper's defences hold up outside the GPT setting.

## What this project does

The study rebuilds the attack and defence setup as LangGraph pipelines and runs it across five models from two providers. It covers two of the paper's attack scenarios, scam and data theft, under different messaging modes and defence configurations. The aim was to test three things: whether the self-replicating infection reproduces outside the GPT family, how much the pipeline's messaging mode changes the spread, and whether the paper's defences actually reduce the attack success rate on these models.

## System design

The attacks run through multi-agent pipelines built in LangGraph, where each agent is an LLM node and the graph controls how messages move between them.

The scam pipeline uses three agents in sequence: Reader, Summariser, and Writer. The Reader ingests a document, the Summariser condenses it, and the Writer produces the final message shown to the user. In a successful attack the Writer outputs the scam content and keeps the injected instruction hidden.

The data theft pipeline uses a different composition: Reader, DB Manager, and Coder. The DB Manager has access to a CSV of fake user records. Under infection it reads that file, appends the data to the propagating prompt, and passes it to the Coder, which writes and runs a POST request to exfiltrate the records.

Two messaging modes were tested. In global mode every agent can see the shared message history. In local mode each agent only sees the output of the agent before it. The two modes change how the infection travels through the graph, which is one of the variables the study measures.

The infection was run in a self-replicating form and a non-replicating baseline, so the effect of the self-copying behaviour could be isolated.

## Models

Five models across two providers:

- Gemini 3.1 Flash Lite (Gemini API)
- Qwen3-27B (Groq)
- LLaMA-3.3-70B (Groq)
- GPT-OSS-120B (Groq)
- GPT-OSS-20B (Groq)

LLaMA-3.3-70B was retired by the provider partway through the combined-defence runs. This is noted in the results where it affects the comparison.

## Defences

The defences from the original paper were implemented and tested against a no-defence baseline: instruction defence, marking, LLM tagging, and delimiting the data. Each was run on its own and then in combination, with the focus on pairing each defence with LLM tagging, following the paper's setup.

## Measuring success

Success is defined per attack type. For the scam pipeline, an attack counts as successful if the final agent outputs the malicious content and hides the infection prompt. For data theft, success means the fake records are exfiltrated through the generated request. Scam outputs are scored with an LLM-as-judge, where a separate model call decides whether the final message is malicious.

## Key findings

The replication holds. Self-replicating infection consistently beats the non-replicating baseline, and the attack works across all five models, so the vulnerability is not specific to the GPT family.

Model behaviour splits by attack type rather than by model size. Qwen3-27B almost fully resists the scam attack but stays highly vulnerable to data theft. GPT-OSS-120B largely resists the scam attack but is fully compromised in the theft pipeline. This matches the original paper's point that a stronger model is not automatically a safer one once a multi-step attack is in play.

Standalone defences are weak. In the scam pipeline, instruction defence, marking, and LLM tagging each cut the attack success rate by about 10 to 12 percentage points from a 34% baseline. Delimiting slightly raised the success rate, so it can be counterproductive on some models. In the theft pipeline, no standalone defence made much difference against a 60% baseline.

Combining defences with LLM tagging helps more. Instruction plus LLM tagging brought theft success down to 36%, and marking plus LLM tagging to 35%. This partly supports the paper's conclusion that defences need to be combined, though the absolute success rates here stay higher than the paper's, which points to real differences in how non-GPT models behave under attack.

## Engineering notes

One bug worth recording. The Writer agent only executed the infection correctly once its selection logic matched on the explicit role name instead of the agent's position in the graph. A positional check silently failed when the pipeline composition changed between attack types, which made the infection look inconsistent until the role-name match was in place.

## Repository structure

Adjust this to match the files you push.

```
.
├── pipelines/        # LangGraph scam and data-theft pipelines
├── defenses/         # instruction, marking, LLM tagging, delimiting
├── evaluation/       # evaluation criteria for each pipeline
├── data/             # synthetic PDFs and the dummy user data CSV
├── results/          # run plots
├── config.py         # model names, API keys, defence list
```

## Running it

Set the provider keys before running:

```bash
export GEMINI_API_KEY=your_key
export GROQ_API_KEY=your_key
```

Install the dependencies (LangGraph and the two provider SDKs), then run the pipeline for the attack type, model, messaging mode, and defence you want to test. The model list and defence settings are defined in the config.
