# RailSwitch: Multimodal Railway Topology Reasoning Benchmark

**RailSwitch** evaluates whether Vision-Language Models (VLMs) perform genuine spatial and topological reasoning over railway track networks rather than relying on clean engineering diagram priors.

## Benchmark Results (Zero-Shot, 5 Topologies, 144 QA Pairs per Model = 288 Total Evaluations)

| Model | Overall Accuracy | Schematic Accuracy | Realistic Overhead Accuracy | Realism Drop | Cross-View Consistency | Avg Batch Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **gemini-3.5-flash-lite** | **95.83%** | 95.83% | 95.83% | 0.00% | 94.44% | 1,937.3 ms |
| **qwen/qwen3.8-27b** | **87.50%** | 88.89% | 86.11% | -2.78% | 91.67% | 743.7 ms |

### Per-Category Accuracy Breakdown
- **Fault Detection:** 100.0% (Gemini) | 100.0% (Qwen)
- **Switch State & Active Track:** 100.0% (Gemini) | 93.75% (Qwen)
- **Counterfactual Reasoning:** 100.0% (Gemini) | 81.25% (Qwen)
- **Grade-Separated Crossing Reasoning:** 100.0% (Gemini) | 83.33% (Qwen)
- **Connectivity:** 93.55% (Gemini) | 90.32% (Qwen)
- **Multi-Hop Route Reasoning:** 90.91% (Gemini) | 72.73% (Qwen)

## End-to-End Architecture
1. **JSON Topology Schema (`data/annotations/topology/`)**: 5 diverse railway layouts (`rail_001` through `rail_005_double_diverted`) covering straight, diverted, multi-switch yards, grade-separated crossings, and structural faults.
2. **Dual-View Renderers (`src/renderer/`)**: Renders paired clean schematics (`data/generated/schematic/`) and realistic overhead aerial views (`data/generated/realistic/`) from the identical underlying graph.
3. **Deterministic Graph Solver (`src/evaluation/ground_truth_generator.py`)**: Computes exact ground-truth reachability, active paths, and fault states.
4. **Question Generator (`src/evaluation/question_generator.py`)**: Synthesizes 72 reasoning questions per view across 6 cognitive categories.
5. **Multi-Model Evaluator (`src/evaluation/model_evaluator.py`)**: Evaluates multimodal models, tracks batch latency, and logs `visual_degradation_failure_cases` in `evaluation_results.json`.

## Quick Start
```powershell
$env:PYTHONPATH="."
$env:GROQ_API_KEY="your_groq_key"
$env:GEMINI_API_KEY="your_gemini_key"
$env:RAILSWITCH_MODELS="qwen/qwen3.8-27b,gemini-3.5-flash-lite"
python run_pipeline.py