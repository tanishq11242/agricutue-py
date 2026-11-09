# agricutue-py
this app is for farmers
 Crop Yield Predictor

A Streamlit app where a farmer enters crop, land, soil, water, fertilizer and farming practices and gets an estimated harvest (total, per acre/hectare, likely range, income) plus tips on what is holding the yield back.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Files

- `app.py`: the farmer-facing form and results page
- `predictor.py`: the estimation logic (no Streamlit needed; run `python predictor.py` for a demo)

## How the estimate works

It is a rule-based agronomic model, not a trained machine-learning model. Each crop has a potential yield on a good farm (quintal per hectare). That is reduced by nine factors: water, fertilizer, soil type, soil pH, sowing time, seed, pests, weeds and weather. The numbers are typical Indian values, so treat the result as a planning estimate.

To make it more accurate for your district, edit the crop table in `predictor.py`, or replace `predict()` with a model trained on real local yield records.
