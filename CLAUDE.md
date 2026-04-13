# Project Rules for Edge AI Satellite Triage Platform (Agentic Edition)

- Language: Python 3.11+
- Framework: PyTorch 2.11+ -> TensorRT for Jetson
- Target: NVIDIA Jetson Orin Nano/AGX/Thor (<20W total inference, measure TOPS/Watt obsessively)
- Agentic layer: Simple ReAct-style reasoning loop (pure Python first). Optional quantized SLM (Phi-3-mini, Gemma-2B, Nemotron-Nano) via ONNX/TensorRT-LLM only on high-value tiles.
- Power rule: CNN inference always; agent loop ONLY if triage_score > 0.6; SLM <5W when used.
- Dual-use: Satellite EO + ground drone/IoT/energy monitoring.
- Style: Modular, type-hinted, zero global state, full docstrings + test stubs.
- Never invent fake datasets — use Sentinel-2 / NASA placeholders.
