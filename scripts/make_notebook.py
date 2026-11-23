"""Builds and executes notebooks/01_eda_and_model_comparison.ipynb."""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

nb = nbf.v4.new_notebook()
C = lambda s: nbf.v4.new_code_cell(s)
M = lambda s: nbf.v4.new_markdown_cell(s)
nb.cells = [
    M("# Vadodara Flood EWS: EDA and model comparison\n**All water levels here are SYNTHETIC-TARGET** (see `data/processed/dataset_card.json`)."),
    C("import sys, json; sys.path.insert(0, '..')\nimport pandas as pd, numpy as np, matplotlib.pyplot as plt\nplt.rcParams['figure.figsize']=(9,3.2)\n"
      "df = pd.read_parquet('../data/processed/flood_dataset.parquet')\ncard = json.load(open('../data/processed/dataset_card.json'))\n"
      "print(card['mode']); print(json.dumps(card['labels'], indent=1)); print(df.shape)"),
    M("## Cleaning report (spikes, gaps, duplicates)"),
    C("pd.DataFrame(card['cleaning_report']).T"),
    M("## Rainfall: annual total and monsoon seasonality"),
    C("r = df[df.station=='akota'].set_index('timestamp')['rain_mmh']\nr.groupby(r.index.year).sum().plot.bar(title='Annual rain (mm), SYNTHETIC')\nplt.show()\n"
      "r.groupby(r.index.month).sum().div(r.index.year.nunique()).plot.bar(title='Mean monthly rain (mm), SYNTHETIC'); plt.show()"),
    M("## Storm example: rain and level at the three stations"),
    C("peak = df.loc[df.level_ratio.idxmax()]; t = peak.timestamp\nw = df[(df.timestamp>t-pd.Timedelta(hours=36))&(df.timestamp<t+pd.Timedelta(hours=24))]\n"
      "fig, ax = plt.subplots(2,1, sharex=True, figsize=(9,5))\nfor s,g in w.groupby('station'): ax[0].plot(g.timestamp, g.level_ratio, label=s)\n"
      "ax[0].axhline(1, color='r', ls='--'); ax[0].set_ylabel('level / danger'); ax[0].legend()\nax[1].bar(w[w.station=='akota'].timestamp, w[w.station=='akota'].rain_mmh, width=0.03); ax[1].set_ylabel('rain mm/h'); plt.show()"),
    M("## Class imbalance: flood-within-3h rate"),
    C("print((df.groupby('station').flood_in_3h.mean()*100).round(2).rename('% rows labelled flood'))\n"
      "df.groupby([df.timestamp.dt.year,'station']).flood_in_3h.sum().unstack().plot.bar(title='flood-labelled hours per year'); plt.show()"),
    M("## Feature correlation with +3 h level change"),
    C("from src.features import FEATURES\nd = df.copy(); d['delta3']=d.level_in_3h-d.level_cm\nd[FEATURES+['delta3']].corr()['delta3'].drop('delta3').sort_values().plot.barh(figsize=(6,8)); plt.show()"),
    M("## Model comparison (TEST, exact values from `models/metrics.json`)"),
    C("m = json.load(open('../models/metrics.json'))\nreg = pd.DataFrame({k:{h:v[h]['mae'] for h in v} for k,v in m['regression_test'].items()}).T.round(3); display(reg)\n"
      "cols=['precision','recall','f1','pr_auc','roc_auc','brier','false_alarm_ratio','detected_events','events','lead_mean_min']\n"
      "display(pd.DataFrame(m['flood_test']).T[cols])"),
    M("## Explainability and error analysis figures"),
    C("from IPython.display import Image, display\nfor f in ['shap_summary','pr_curves','reliability','test_event_forecast']: display(Image(f'../docs/figures/{f}.png'))"),
]
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": "notebooks/"}})
nbf.write(nb, "notebooks/01_eda_and_model_comparison.ipynb")
print("notebook executed")
