# Pipeline Run Summary

## Summary
- pipeline_id: pipeline
- config: /tmp/pytest-of-kyrie/pytest-16/test_run_pipeline_branches0/pipeline.yaml
- run_dir: .
- started_at: 2026-02-06T14:58:54.706614
- finished_at: 2026-02-06T14:58:54.706631

## dataset
- python: python
- command: gen.py

## train
- python: /usr/bin/python
- command: train.py --output-dir out_train --flag --data-root out

## eval
- python: /usr/bin/python
- command: eval.py --flag --data-root out --output-dir out_train/eval
