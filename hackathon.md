# OpenEnv Hackathon — Participant Guide

## Rules & Requirements

- **Open Source:** Your repository must be public.
- **New Work Only:** All projects must be started from scratch during the hackathon — no previous work permitted.
- **Team Size:** Maximum of **2** members per team.
- **Disqualification:** Projects will be disqualified for violating legal, ethical, or platform policies, or for using code, data, or assets you do not have rights to.
- Your project **must** use **OpenEnv (stable release 0.2.1)** deployed on **HF Spaces**.
- You must show a **minimal training script** for your environment using **Unsloth or HF TRL** in Colab.
- You must upload a **one-minute demo video** to YouTube as part of your submission.

---

## Problem Statements

Your project must address **at least one** of the five problem statements below.

- You may align with **multiple partner sub-problem statements**, but can only be **judged for a maximum of two**.
- Partner sub-problem statements are eligible for **extra prizes of $10,000 USD each**, judged separately from the main track.

---

### Statement 1: Multi-Agent Interactions

Environments for cooperation, competition, negotiation, and coalition formation. Drives theory-of-mind reasoning and emergent strategic behavior.

- **Expected Outcome:** An environment that can be used to train multi-agent task handling in an LLM.
- **Example Environments:** Market simulations, compute-allocation negotiations, collaborative puzzle worlds, mixed cooperative/competitive strategy games.

**Partner Sub-Themes:**
- **Fleet AI:** *Scalable Oversight* — Environments that train oversight agents to monitor, analyze, and explain the behavior of other AI agents in complex, multi-agent settings.
- **Halluminate:** *Multi-Actor Environments* — A realistic environment where an agent interacts with and manages multiple actors to discover and achieve a task.

---

### Statement 2: (Super) Long-Horizon Planning & Instruction Following

Environments requiring deep, multi-step reasoning with sparse or delayed rewards. Goal: enable agents to decompose goals, track state over extended trajectories, and recover from early mistakes.

- **Expected Outcome:** An environment that captures and improves LLM behavior on challenging long-horizon tasks beyond context memory limits.
- **Example Environments:** Research-planning simulators, large-scale codebase refactoring, strategic resource management, long-horizon logistics optimization, extremely complex instruction following (e.g., 300 scattered instructions).

**Partner Sub-Themes:**
- **Mercor:** Make an environment with capped/uncapped rewards where frontier model rewards scale with token output.
- **Scale AI:** Environments for long-horizon workflows for non-code business use cases — focusing on Sales, Project Management, or HR & IT.

---

### Statement 3: World Modeling

#### 3.1 Professional Tasks
Environments requiring real interaction with tools, APIs, or dynamic systems. Goal: strengthen causal reasoning and persistent world models.

- **Expected Outcome:** An environment capturing nuances of a defined partially observable world and improving LLM interaction with it.
- **Example Environments:** Dynamic browser/API ecosystems, enterprise applications, scientific workflow loops, economic simulations with feedback, tool-discovery benchmarks.

**Partner Sub-Theme:**
- **Scaler AI Labs:** *Multi-App RL Environment for Enterprise Workflows* — RL environments demonstrating complex workflows and business rule nuances in a large enterprise.

#### 3.2 Personalized Tasks
Environments for real personalized task handling — personal messages, scheduling conflicts, tough emails, personal assistant tasks.

- **Expected Outcome:** An environment that gives the model a realistic simulation of handling personal tasks, conflicts, and managing them as delegations.
- **Example Environments:** Executive assistant meeting planner, dinner and travel planning, email and message replying.

**Partner Sub-Theme:**
- **Patronus AI:** *Consumer Workflows with Schema Drift* — Multi-step consumer workflow environments where underlying data schemas, API contracts, and policies/rules change over time.

---

### Statement 4: Self-Improvement

Environments where agents learn to generate new challenges, escalate difficulty, and improve through self-play or adaptive curricula. Goal: recursive skill amplification.

- **Expected Outcome:** An environment for improving self-play of an LLM over a defined set of tasks.
- **Example Environments:** Self-play negotiation arenas, auto-generated math/proof tasks, evolving coding competitions, adaptive RL curricula.

**Partner Sub-Theme:**
- **Snorkel AI:** *Simulated Experts-in-the-Loop* — Environment that simulates interactions with real subject-matter experts, with changing requirements and preferences.

---

### Statement 5: Wild Card — Impress Us!

Doesn't fit the boxes above? Be creative. Projects must meaningfully add value to LLM training on a defined task.

---

## Submission

Submit at: **https://cerebralvalley.ai/e/openenv-hackathon-sf/hackathon/submit**

Required with submission:
- Public GitHub repository (new work only)
- One-minute YouTube demo video
- Minimal training script (Unsloth or HF TRL) in Colab
- OpenEnv (v0.2.1) deployed on HF Spaces
- Selection of **up to 2** partner sub-tracks (if applicable)

---

## Judging Criteria

| Criteria | Weight |
|---|---|
| Environment Innovation — novelty, creativity, challenge | 40% |
| Storytelling — clarity of problem, environment, and agent behavior | 30% |
| Training Script Showing Improvement in Rewards | 20% |
| Reward and Training Pipeline Setup | 10% |

**Judging Process:**
1. **Round 1:** All teams pitch to assigned judges — ~3 min pitch + 1–2 min Q&A.
2. **Round 2 (Finals):** Top **6** teams demo on stage — ~3 min pitch + 2–3 min Q&A.

> Do **not** show a slide presentation. Show us what you built. Judges will verify the project was built entirely during the event.

---

## Prizes

| Place | Prize |
|---|---|
| 🥇 1st Place | $15,000 USD |
| 🥈 2nd Place | $9,000 USD |
| 🥉 3rd Place | $6,000 USD |
| Partner Sub-Track (each) | $10,000 USD |

---

## Resources

- **OpenEnv Docs & Architecture:** See provided slideshow (fundamentals, local dev, Docker, HF Spaces deployment, training with TRL & Unsloth, GPU request form).
- **Unsloth AI:** https://unsloth.ai/docs/get-started/unsloth-notebooks#grpo-reasoning-rl-notebooks
- **Mercor Dataset:** https://huggingface.co/datasets/mercor/apex-agents
- **Mercor Archipelago Repo:** https://github.com/Mercor-Intelligence/archipelago
- **HuggingFace Credits ($30):** https://huggingface.co/openenv-community