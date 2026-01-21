# AI Architect 🤖🧠

A **multi-agent AI system** that helps users design **GPU infrastructure, model choices, and Kubernetes-based architectures** for AI/ML workloads.

This project acts like an **AI solutions architect**: it asks the *right clarification questions*, extracts structured requirements, retrieves evidence (GPUs, models, infra), and produces **grounded, explainable recommendations**.

---

## ✨ Key Features

* **Multi-agent orchestration** (Clarifier → Extractor → Retriever → Final Architect)
* **Strict schema-driven reasoning** (JSON-first, tool-safe outputs)
* **GPU & infrastructure recommendations** (cloud / Kubernetes focused)
* **Model recommendations** with VRAM and task constraints
* **Compliance-aware design** (e.g. GDPR, data residency)
* **Mermaid architecture diagrams** generated automatically

---

## 🧩 Architecture Overview

The system is intentionally modular and agent-based:

1. **Clarifying Agent**
   Asks minimal, high-signal questions only when required.

2. **Requirements Extractor Agent**
   Produces a decision-ready `requirements.json` with *no assumptions*.

3. **Retrieval Planner Agent**
   Decides whether external evidence (GPUs, models, infra) is needed.

4. **Final Architect Agent**
   Generates grounded recommendations: GPUs, models, Kubernetes components, cost estimates, and diagrams.

This design avoids hallucination by **gating answers on evidence availability**.

---



## ⚙️ Tech Stack

* **Python**
* **LLMs**: Mistral / OpenAI (configurable)
* **Agent frameworks**: LangGraph-style orchestration
* **Schemas**: Strict JSON / Pydantic-style validation
* **Diagrams**: Mermaid

---



## 🧪 Example Use Cases

* "I have 2TB of medical MRI data and want to fine-tune a segmentation model"
* "Recommend GPUs and Kubernetes setup for LLM fine-tuning"
* "Design a compliant AI pipeline for sensitive data in the EU"

The system will:

* Ask only *necessary* questions
* Refuse to answer if evidence is missing
* Output **copy-paste-ready JSON**

---

## 🛡️ Design Philosophy

* ❌ No guessing

* ❌ No silent assumptions

* ❌ No hallucinated prices or GPUs

* ✅ Evidence-gated reasoning

* ✅ Explainable architecture decisions

* ✅ Production-oriented thinking

This is **not** a chatbot — it is an **AI architect**.

---

## 📌 Status

This project is **actively developed** and used as:

* MSc-level applied AI work
* Portfolio-grade agentic system
* Foundation for AI infrastructure advisory tools

---


## 👤 Author

Built by **Mojdeh Ghadami**
MSc Artificial Intelligence & Data Science
Focus: Agentic AI systems, LLM orchestration, AI infrastructure design

