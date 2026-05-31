# ADR 002: EfficientNet-B3 Backbone

## Status
Accepted

## Context
Need efficient, accurate backbone for 224x224 images.

## Decision
Use EfficientNet-B3 (12M params).

## Consequences
- Good accuracy/speed tradeoff
- Pretrained on ImageNet
- Easy to unfreeze progressively
