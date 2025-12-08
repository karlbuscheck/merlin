# Merlin — What-If Engine for YouTube Creators

Merlin is a forecasting engine that predicts YouTube video performance **before** an episode goes live. It takes a title, episode length, and simple scheduling metadata (day of week, month) and returns **predicted lifetime views and subscriber change**, so creators can A/B test ideas with real data instead of gut feel.

## Live Demo — Merlin v1 (Streamlit Web App)

**Try Merlin in your browser:**
[Open the Merlin v1 app](https://merlin-mzuorffplqskcrvhe2dkds.streamlit.app/)

*Note: If the app has been idle, Streamlit may show a brief “waking up” / “in the oven” screen while it starts the container. Wait a few seconds, and the interface will load.*

**What you can do in v1:**

- Enter a **video title**, **duration**, **day of week**, and **month**
- Get **predicted lifetime views** and **predicted subscriber change**
- Quickly test alternate titles and runtimes for the same episode

Under the hood, Merlin v1 uses:

- A **MiniLM sentence transformer** to embed titles
- **XGBoost regressors** to predict views and subscriber change
- A lightweight **Streamlit** front-end deployed on Streamlit Cloud

# Merlin v0 (Notebook Prototype)

Merlin is a what-if engine for predicting YouTube video performance *before* an episode goes live. Originally built for a daily sports show, it helps creators compare alternate title ideas and structural choices using real data instead of instinct. The model predicts each episode’s **lifetime views** and **subscriber change** from its **title**, **episode length**, and lightweight metadata (day-of-week, month).

The v0 model is trained on a full year of proprietary performance data from the daily sports show I work on, giving it production-grade grounding. While this version is fine-tuned for that show, the pipeline itself is modular — any YouTube creator with historical data can retrain Merlin on their own channel to get a tailored forecasting engine.

## Problem & Context

Most YouTube decisions — title, length, timing — happen before a video goes live, yet creators have no way to A/B test options in advance. Merlin provides a lightweight forecasting layer that predicts views and subscriber change from simple inputs, enabling quick, real-time comparisons before publishing.

## Technical Highlights

**Hybrid model architecture**
- Title embeddings generated with the `all-MiniLM-L6-v2` sentence-transformer  
- XGBoost regressors for predicting lifetime views and subscriber change  
- Feature set includes title embeddings, episode length, day-of-week, and month.

**Log-transform rescue for views**
- Lifetime views were highly skewed, making raw-target modeling unstable  
- Switching to a `log1p` target gave the model a learnable signal and improved generalization on high-variance episodes  
- After reversing with `expm1`, the model achieved a **~377-view median absolute error** on held-out episodes — turning a noisy baseline into a practical forecasting tool  

**Robust subscriber modeling**
- Subscriber change includes negative dips, flat outcomes, and rare spikes  
- Using `RobustScaler` helped reduce outlier influence before XGBoost training  
- Produces directionally meaningful subscriber-change predictions for real title testing  

**Reproducible pipeline**
- Training workflow is split across clear notebooks: data prep → embedding generation → model training → artifact serialization  
- Inference is fully decoupled: Notebook 05 loads all saved models, feature schemas, and metadata to deliver fast predictions  
- Merlin Console offers a simple, interactive prompt-based interface for real-time what-if analysis

## Project Structure

```text
merlin/
├── README.md
├── models/
│   ├── xgb_views_booster.json
│   ├── xgb_subs_model.pkl
│   ├── xgb_subs_target_scaler.pkl
│   ├── features_views.json
│   ├── features_subs.json
│   └── (model metadata files)
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_title_embeddings.ipynb
│   ├── 03a_model_training_views_xgboost.ipynb
│   ├── 03b_model_training_subscribers_xgboost.ipynb
│   ├── 04_streamlit_app.ipynb
│   └── 05_merlin_console.ipynb
└── data/
    └── (raw + processed CSVs used in training)
```
## How to Run Merlin v0 in Colab

1. Open the notebook: `notebooks/05_merlin_console.ipynb`
2. Run all cells once to mount Drive, load the trained boosters, and define the helper functions. The final cell lauches the interactive console:
   `run_merlin_console()`

3. When prompted, enter:
   - title  
   - episode length (minutes)  
   - day of week  
   - month  

Merlin will output predicted lifetime views and predicted subscriber change.
<img width="1129" height="487" alt="Screenshot 2025-11-25 at 10 27 41 PM" src="https://github.com/user-attachments/assets/b5660efd-ac76-40e4-a9ab-9cfa25cb5e5e" />

## Next Steps

- **Deploy to Streamlit Cloud**  
  Turn Merlin into an interactive web app so creators can run real-time A/B tests.

- **Integrate thumbnail embeddings**  
  Add vision features using ViT or CLIP to incorporate thumbnail quality into forecasts.

- **RAG-powered versioning**  
  Use Retrieval-Augmented Generation to analyze historical show notes, metadata, and title styles.

The Streamlit deployment will mark Merlin v1, the first fully interactive version of the tool. Adding thumbnail embeddings becomes v1.5, bringing Merlin into multimodal forecasting. And the RAG-powered system is v2, where Merlin starts to understand long-term creator patterns, not just a single title.
