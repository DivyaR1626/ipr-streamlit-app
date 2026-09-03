# ipr-streamlit-app
# IPR Curve Analysis App

An interactive Streamlit application for generating **Inflow Performance Relationship (IPR)** curves using three industry-standard methods — Simple (Constant J), Vogel, and Fetkovich — with support for future IPR prediction, real field data, and custom well test uploads.

## What is IPR?

The Inflow Performance Relationship describes how much oil a well can produce (Qo) at a given flowing bottomhole pressure (Pwf). It's a core tool in petroleum production engineering used for well performance evaluation, nodal analysis, and forecasting.

## Features

- **Three IPR methods**
  - **Simple / Constant J** — single stabilized test point, linear IPR
  - **Vogel** — for saturated and undersaturated reservoirs (with bubble-point handling)
  - **Fetkovich** — multi-rate test regression (log-log plot with n, C coefficients)
- **Future IPR prediction** — project how the IPR curve shifts as reservoir pressure declines
- **Real field data demos**
  - A real published multi-rate test (Well B, Keokuk Pool, Oklahoma, 1935) — includes an actual repeat test 8 months later, used to validate the future-IPR prediction against real data
- **Custom data upload** — bring your own well test data (CSV/Excel)
- **AOF (Absolute Open Flow) reporting** — present and future, with % change
- **Downloadable results** — generated IPR curve values as CSV

## Try it live

## Running locally

The app will open at `http://localhost:8501`.

## Running in GitHub Codespaces

Click the **Open in GitHub Codespaces** badge above — the container builds automatically, installs all dependencies from `requirements.txt`, and launches the app for you. No local setup required.

## Project structure

```
.
├── .devcontainer/
│   └── devcontainer.json      # Codespaces / VS Code dev container config
├── PTAapp.py                  # Main Streamlit application
├── requirements.txt           # Python dependencies
└── README.md
```

## Tech stack

- [Streamlit](https://streamlit.io/) — web app framework
- [Plotly](https://plotly.com/python/) — interactive IPR plotting
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling and calculations

## Methods reference

| Method | Best for | Inputs required |
|---|---|---|
| Simple (Constant J) | Undersaturated reservoirs, single stabilized test | Pr, one (Qo, Pwf) test point |
| Vogel | Saturated / two-phase flow reservoirs | Pr, one (Qo, Pwf) test point |
| Fetkovich | Multi-rate tests, general applicability | Pr, multiple (Qo, Pwf) test points |

## Roadmap

- [ ] TPR (Tubing Performance Relation) curve overlay for nodal analysis
- [ ] Theoretical (Darcy) productivity index comparison using reservoir/well parameters.
