# EO-Agents? A Three-Agent LLM Pipeline for Earth-Observation Hypothesis Generation

An end-to-end pipeline for discovering plausible, unused combinations of
NASA Earth-observation datasets, combining:

1. **A heterogeneous Graph Neural Network** trained on the NASA EO
   Knowledge Graph to rank novel (dataset, dataset) pairs by predicted
   co-usage likelihood.
2. **A three-agent LLM pipeline** that (a) scores pair plausibility and
   novelty, (b) generates a structured research hypothesis for the pair,
   and (c) independently judges the hypothesis on importance,
   tractability, and novelty.

We evaluate the pipeline in a 2 x 2 x 2 factorial design where each of
the three agents is either GPT-5.2 or Claude Sonnet 4.6, producing 8
experimental conditions and 640 validator judgments over 160 generated
hypotheses.

## Repository structure

    pipeline/          Core pipeline scripts, one per stage
      stage0_pipeline.py       data pipeline, temporal split
      stage1_baselines.py      6 link-prediction baselines
      stage2_gnn.py            main GNN (homo + heterogeneous)
      stage2p5_ablation.py     11-lever cumulative ablation
      stage2p5b_verify.py      multi-seed verification
      stage2p5c_save_final.py  save final embeddings
      stage4_build_strata.py   construct 4 strata for LLM judgment
      stage4_judge_submit.py   submit OpenAI batch for Stage 4
      stage4_judge_parse.py    parse Stage 4 batch results
      stage4_hero_pairs.py     tier analysis of predicted-novel pairs
      stage4_analyze.py        aggregate Stage 4 results
      stage5_factorial.py      2x2x2 factorial pipeline
      stage5_analyze.py        factorial analysis (variance, inter-rater, flagship)

    figures/           Publication figures and tables
      fig_pipeline.py              Figure 1: pipeline schematic
      fig_stage4_judgment.py       Figure 2: Stage 4 judgments per stratum
      fig_stage5_factorial.py      Figures 3-5: factorial results
      gen_tables.py                LaTeX tables (booktabs)

    logs/              Run logs from the experiments reported in the paper

## Setup

    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt

Set API keys:

    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...

## Data

The NASA EO Knowledge Graph is downloaded separately from:

  https://huggingface.co/datasets/nasa-gesdisc/nasa-eo-knowledge-graph

Place `graph.graphml` under `./nasa_eo_kg/`.

## Reproducing the pipeline

All scripts are run from the repo root. Each stage writes its outputs to
`./neurips_figs/<stage>/` and picks up its inputs from earlier stages.

Stages 1-2: graph learning

    python3 pipeline/stage0_pipeline.py
    python3 pipeline/stage1_baselines.py
    python3 pipeline/stage2_gnn.py
    python3 pipeline/stage2p5_ablation.py
    python3 pipeline/stage2p5b_verify.py
    python3 pipeline/stage2p5c_save_final.py

Stage 4: pair-level LLM judgment (GPT batch API)

    python3 pipeline/stage4_build_strata.py
    python3 pipeline/stage4_judge_submit.py --live --stratum B   # repeat for A, C, D
    python3 pipeline/stage4_judge_parse.py                        # after batches complete

Stage 5: 2x2x2 factorial hypothesis generation + validation

    python3 pipeline/stage5_factorial.py --task agent1_gpt       --live
    python3 pipeline/stage5_factorial.py --task agent1_claude    --live
    python3 pipeline/stage5_factorial.py --task select_top40     --live
    python3 pipeline/stage5_factorial.py --task generate         --live
    python3 pipeline/stage5_factorial.py --task validate         --live
    python3 pipeline/stage5_factorial.py --task analyze
    python3 pipeline/stage5_analyze.py

Figures and tables

    python3 figures/fig_pipeline.py
    python3 figures/fig_stage4_judgment.py
    python3 figures/fig_stage5_factorial.py
    python3 figures/gen_tables.py

## Cost

Approximate OpenAI + Anthropic API cost for the full experiment: ~$20.

## Citation

(to be added on paper acceptance)

## License

(to be added)
