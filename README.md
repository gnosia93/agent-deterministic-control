# Agent Deterministic Control and System Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

This repository is dedicated to exploring how to build **production-ready, deterministic LLM Agents**. Moving beyond simple single-turn chatbots, we dive deep into the system architectures, tracing, and state-machine constraints required to tame the inherent non-determinism of Large Language Models.

## 🎯 Key Challenges We Solve
* **The Chaos of Non-Determinism:** Preventing agents from wandering off or picking random, suboptimal tools.
* **Infinite Loop Mitigation:** Implementing strict fail-safes and maximum loop thresholds.
* **Granular Tracing & Debugging:** Moving away from line-by-line debugging to event-driven tracing (using LangSmith/Langfuse).


## 🛠️ Repository Structure (Planned)

* `01-raw-api-loop/`: Implementing the ReAct loop from scratch using pure OpenAI/Anthropic APIs without frameworks (Reverse Engineering).
* `02-state-machine-constraints/`: Building deterministic workflows using **LangGraph** / **CrewAI**.
* `03-llmops-tracing/`: Hands-on configurations for **LangSmith** and **Langfuse** integration.
* `04-regression-testing/`: Setting up LLM-as-a-judge automated evaluation pipelines.




## 📚 Recommended Readings & References

* 🥇 《AI Engineering》 (또는 《Designing Machine Learning Systems》) — Chip Huyen 저
* 🥈 《Build a Large Language Model (From Scratch)》 — Sebastian Raschka 저
* "A practical guide to building agents" (OpenAI Business Guide)
  *	OpenAI가 공식 발행한 에이전트 구축 가이드입니다. 에이전트의 워크플로우를 무조건 자율로 두지 말고, 뼈대(Scaffolding)를 세워 제어하는 '프롬프트 인젝션 방어 기법'과 '결정적 흐름(Deterministic flows)' 설계의 정석을 다룹니다.
* [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
* [《ReAct: Synergistic Reasoning and Acting in Language Models》](https://arxiv.org/abs/2210.03629)
* 《Toolformer: Language Models Can Teach Themselves to Use Tools》 (Schick et al.)
  * LLM이 어떻게 문맥을 '생각(Reasoning)'하고 '행동(Action)'으로 연결하는지 언어학적/컴퓨터공학적 원리를 처음 세상에 증명한 논문들입니다. 수학식이 많지 않고 개념 위주라 개발자가 읽기에 아주 좋습니다.

