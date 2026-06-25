# Agent Deterministic Control and System Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

This repository is dedicated to exploring how to build **production-ready, deterministic LLM Agents**. Moving beyond simple single-turn chatbots, we dive deep into the system architectures, tracing, and state-machine constraints required to tame the inherent non-determinism of Large Language Models.

## 🎯 Key Challenges We Solve
* **The Chaos of Non-Determinism:** Preventing agents from wandering off or picking random, suboptimal tools.
* **Infinite Loop Mitigation:** Implementing strict fail-safes and maximum loop thresholds.
* **Granular Tracing & Debugging:** Moving away from line-by-line debugging to event-driven tracing (using LangSmith/Langfuse).

---

## 🏗️ Core Architecture Pattern

Our approach transitions the agent from complete autonomy to a structured **State Machine (Graph)** topology.














```
 랭체인 공식 문서 예제를 복사해서 붙여넣거나 타임라인에 돌아다니는 간단한 챗봇 코딩만 반복해서는 결코 이 구조를 꿰뚫어 볼 수 없습니다.
내부의 수학적 원리, 추론 엔진의 작동 방식, 그리고 비결정성(Non-deterministic)을 제어하는 "엔지니어링 아키텍처" 관점으로 접근해야 비로소 에이전트를 내 손으로 완벽하게 통제할 수 있습니다.
단순 '사용법'을 넘어 인터널(Internals)과 아키텍처 정석을 깊게 파고들 수 있는 최고의 추천 자료 및 책을 엄선해 드립니다.
📚 1. 추천 도서 (The Books)
현재 글로벌 AI 엔지니어들 사이에서 "프레임워크 사용법이 아닌 시스템 설계와 원리를 다루는 최고의 명저"로 꼽히는 책들입니다.
🥇 《AI Engineering》 (또는 《Designing Machine Learning Systems》) — Chip Huyen 저
•	왜 봐야 하는가: 실리콘밸리에서 가장 영향력 있는 ML 엔지니어인 칩 후옌의 신작과 전작입니다. LLM 기반 애플리케이션의 고질적인 문제인 Inference, Evaluation, Monitoring, 그리고 예측 불가능성을 프로덕션 레벨에서 어떻게 제어할 것인가를 시스템 디자인 패턴 관점에서 집요하게 다룹니다.
•	핵심 키워드: 비결정성 제어, 스트리밍 인퍼런스, 실시간 평가 아키텍처.
🥈 《Build a Large Language Model (From Scratch)》 — Sebastian Raschka 저
•	왜 봐야 하는가: 에이전트가 왜 그런 판단을 하는지 이해하려면 결국 트랜스포머(Transformer)와 어텐션(Attention) 메커니즘의 밑바닥을 알아야 합니다. 파이토치(PyTorch) 코드로 LLM의 뇌를 직접 구현해보면서 토큰 확률 배분이 어떻게 일어나는지 이해하게 되며, 이를 알고 나면 "아, 프롬프트의 이 단어 때문에 툴 콜링 확률 궤적이 이렇게 꺾였구나"가 직관적으로 보이기 시작합니다.
•	핵심 키워드: 토큰 임베딩, 어텐션 가중치, 인퍼런스 인터널.
🥉 《LLMs in Production》 — Christopher Brousseau & Matthew Sharp (Manning Publications)
•	왜 봐야 하는가: 실험실(Lab)에서 잘 돌던 에이전트가 쿠버네티스(EKS)나 프로덕션 환경으로 나갔을 때 마주하는 현실적인 문제(비용 최적화, 로드 테스트, 리트라이 밸런싱, MLOps 파이프라인)를 다룹니다. 에이전트가 오작동할 때의 세이프넷(Safety net) 구축법을 아주 체계적으로 배울 수 있습니다.
🌐 2. 반드시 정독해야 할 최고 권위의 기술 블로그 & 논문
책은 출판되는 데 시간이 걸리지만, 에이전트 아키텍처의 정석은 주요 AI 기업들의 기술 백서와 논문에 다 들어있습니다.
① OpenAI 공식 기술 가이드라인 & Cookbook
•	추천 자료: "A practical guide to building agents" (OpenAI Business Guide)
•	내용: OpenAI가 공식 발행한 에이전트 구축 가이드입니다. 에이전트의 워크플로우를 무조건 자율로 두지 말고, 뼈대(Scaffolding)를 세워 제어하는 '프롬프트 인젝션 방어 기법'과 '결정적 흐름(Deterministic flows)' 설계의 정석을 다룹니다.
② Anthropic (Claude) 기술 블로그
•	추천 아티클: "Building Effective Agents" (Anthropic 리서치 블로그)
•	내용: 랭체인 없이 순수하게 코드로 에이전트를 제어하는 "컴포저블 아키텍처(Composable Architecture)" 패턴을 정립한 최고의 글입니다. Augmented LLM, Workflow(Chain/Routing/Parallelization), Agent(ReAct)의 차이점을 명확히 쪼개어 설명해주어 뇌가 맑아지는 느낌을 줍니다.
③ 에이전트 논문의 시초 리딩
•	ReAct 논문: 《ReAct: Synergistic Reasoning and Acting in Language Models》 (Yao et al.)
•	Toolformer 논문: 《Toolformer: Language Models Can Teach Themselves to Use Tools》 (Schick et al.)
•	내용: LLM이 어떻게 문맥을 '생각(Reasoning)'하고 '행동(Action)'으로 연결하는지 언어학적/컴퓨터공학적 원리를 처음 세상에 증명한 논문들입니다. 수학식이 많지 않고 개념 위주라 개발자가 읽기에 아주 좋습니다.
🛠️ 3. 학습을 위한 올바른 접근 순서 (How to Study)
여러 번 코딩해도 정석을 인지하기 어려웠던 이유는 프레임워크(LangChain)가 원리를 너무 꽁꽁 숨겨두었기 때문입니다. 아래 순서로 역공학(Reverse Engineering) 공부를 해보시는 것을 추천합니다.
	1.	원시 코드로 짜보기 (No Framework): 랭체인을 완전히 걷어내고, 오직 requests 라이브러리로 OpenAI/Anthropic raw API만 호출해서 while 루프 문으로 직접 Tool 호출 신호를 파싱하고 수동으로 파썬 함수를 매핑하는 코드를 딱 한 번만 짜보세요. (앞서 나눈 대화의 내부 통신 구조 JSON을 내 손으로 파싱해보는 것입니다.)
	2.	트레이싱 툴로 현미경 디버깅: 코드를 돌릴 때 LangSmith나 Langfuse를 무조건 연동해 두고, 프롬프트 글자 한 줄 바꿨을 때 전체 토큰 그래프와 입력/출력이 어떻게 연쇄 반응을 일으키는지 로그를 집요하게 뜯어보는 훈련을 하셔야 합니다.
이 관점으로 접근하시기 시작하면, 앞으로 다가올 대규모 AI 인프라(Inference on EKS) 위에서 에이전트 서비스가 왜 네트워크 병목을 일으키고, 이를 제어하기 위해 백엔드 구조를 어떻게 잡아야 하는지 눈이 완전히 뜨이실 겁니다.
```
