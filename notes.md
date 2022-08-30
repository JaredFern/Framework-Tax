# Hardware Aware Architecture Design

## What is the goal of this work? What's the story?

1. Demonstrate that "efficient" architectures are not a catch-all term
2. Improve model and architecture selection for downstream deployment
   1. Develop a metric for model latency solely based on architecture and deployment hardware
3. Model Design Proxy so that

## What are the current results?

1. Model latency is dependent on both device and model parallelism
2. Scaled linear layers can be used as a best-case baseline for hardware performance

## What is needed to demonstrate our goal?

## Strengths

1. Diversity of model architectures and hardware platforms
2.

## Current Concerns

1. Latency is not the only concern in deployment -- ignoring performance, memory usage, etc.
2. Latency prediction can be done with less features
3. Not an exhaustive hardware search --> Needs to be model-first work
4. Want to avoid diving into compiled operations and hardware-level blocks
