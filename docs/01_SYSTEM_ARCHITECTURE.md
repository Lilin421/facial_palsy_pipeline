# System Overview

## Goal

Automatic facial palsy assessment using

- Audio
- Landmark
- Video VLM
- Clinical LLM

---

## Scope

This project aims to

- locate facial movements
- generate weak action priors
- guide a Video VLM
- assist clinical grading

The project does not

- replace clinicians
- diagnose directly
- perform grading inside the landmark module

---

## Pipeline

Video

↓

Audio

↓

Landmarks

↓

Motion Analysis

↓

Temporal Localization

↓

Evidence Fusion

↓

Video VLM

↓

Clinical LLM

## Landmark Module

Responsible for

- tracking
- temporal localization
- action hypotheses

Not responsible for

- diagnosis
- grading
- appearance verification
