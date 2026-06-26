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
  * LLM 기반 애플리케이션의 고질적인 문제인 Inference, Evaluation, Monitoring, 그리고 예측 불가능성을 프로덕션 레벨에서 어떻게 제어할 것인가를 시스템 디자인 패턴 관점에서 집요하게 다룹니다 - 비결정성 제어, 스트리밍 인퍼런스, 실시간 평가 아키텍처.
* 🥈 《Build a Large Language Model (From Scratch)》 — Sebastian Raschka 저
  * 왜 봐야 하는가: 에이전트가 왜 그런 판단을 하는지 이해하려면 결국 트랜스포머(Transformer)와 어텐션(Attention) 메커니즘의 밑바닥을 알아야 합니다. 파이토치(PyTorch) 코드로 LLM의 뇌를 직접 구현해보면서 토큰 확률 배분이 어떻게 일어나는지 이해하게 되며, 이를 알고 나면 "아, 프롬프트의 이 단어 때문에 툴 콜링 확률 궤적이 이렇게 꺾였구나"가 직관적으로 보이기 시작합니다.
  *	핵심 키워드: 토큰 임베딩, 어텐션 가중치, 인퍼런스 인터널.
* 🥉 《LLMs in Production》 — Christopher Brousseau & Matthew Sharp (Manning Publications)
  *	왜 봐야 하는가: 실험실(Lab)에서 잘 돌던 에이전트가 쿠버네티스(EKS)나 프로덕션 환경으로 나갔을 때 마주하는 현실적인 문제(비용 최적화, 로드 테스트, 리트라이 밸런싱, MLOps 파이프라인)를 다룹니다. 에이전트가 오작동할 때의 세이프넷(Safety net) 구축법을 아주 체계적으로 배울 수 있습니다.

#### 🌐 반드시 정독해야 할 최고 권위의 기술 블로그 & 논문 ####

* "A practical guide to building agents" (OpenAI Business Guide)
  *	OpenAI가 공식 발행한 에이전트 구축 가이드입니다. 에이전트의 워크플로우를 무조건 자율로 두지 말고, 뼈대(Scaffolding)를 세워 제어하는 '프롬프트 인젝션 방어 기법'과 '결정적 흐름(Deterministic flows)' 설계의 정석을 다룹니다.
* "Building Effective Agents" (Anthropic 리서치 블로그)
  *	내용: 랭체인 없이 순수하게 코드로 에이전트를 제어하는 "컴포저블 아키텍처(Composable Architecture)" 패턴을 정립한 최고의 글입니다. Augmented LLM, Workflow(Chain/Routing/Parallelization), Agent(ReAct)의 차이점을 명확히 쪼개어 설명해주어 뇌가 맑아지는 느낌을 줍니다.
* 《ReAct: Synergistic Reasoning and Acting in Language Models》 (Yao et al.)
* 《Toolformer: Language Models Can Teach Themselves to Use Tools》 (Schick et al.)
  * LLM이 어떻게 문맥을 '생각(Reasoning)'하고 '행동(Action)'으로 연결하는지 언어학적/컴퓨터공학적 원리를 처음 세상에 증명한 논문들입니다. 수학식이 많지 않고 개념 위주라 개발자가 읽기에 아주 좋습니다.

