---
name: Bug Report
about: Report a reproducible bug in training, evaluation, inference, or the Slicer plugin
title: "[BUG] <Short description>"
labels: ["bug", "needs-triage"]
assignees: []
---

## Bug Description

<!-- A clear, concise description of the problem. -->

## Steps to Reproduce

<!-- Minimum steps to reproduce the behaviour consistently. -->

1. 
2. 
3. 

## Expected Behaviour

<!-- What did you expect to happen? -->

## Actual Behaviour

<!-- What actually happened? Include error messages verbatim. -->

## Error Output

```
Paste the full traceback / error message here.
```

## Environment

<!-- Run `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` -->

| Field | Value |
|---|---|
| OS | <!-- e.g. Ubuntu 22.04, Windows 11 --> |
| Python version | <!-- e.g. 3.11.8 --> |
| PyTorch version | <!-- e.g. 2.3.0 + cuda 12.1 --> |
| SAM2 installed? | <!-- yes / no / which branch --> |
| GPU | <!-- e.g. NVIDIA A100 80 GB / CPU only --> |
| CUDA version | <!-- e.g. 12.1 --> |
| Package version / commit | <!-- git log --oneline -1 --> |

## Minimal Reproducible Example

<!-- Smallest self-contained script that triggers the bug. -->

```python
# Paste code here
```

## Config File (if training/evaluation)

```yaml
# Paste the relevant section of your config here
```

## Additional Context

<!-- Screenshots, log files, dataset subset, anything else. -->

<!-- 
Checklist before submitting:
  [ ] I searched existing issues and this is not a duplicate
  [ ] I am using the latest commit on `main`
  [ ] I can reproduce this consistently
  [ ] I included the full error traceback above
-->
