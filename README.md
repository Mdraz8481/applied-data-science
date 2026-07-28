# Data Analysis Portfolio

Five self-contained analyses in Python — statistical inference, exploratory data analysis,
resampling, natural language processing, and unsupervised machine learning. Each project
states a question up front, shows the work, and ends with findings and their limitations.

Every notebook runs top to bottom from a clean checkout (see [Setup](#setup)). Projects 02
and 05 use datasets that aren't redistributable; each has a `data/README.md` with a
one-line download.

---

## Projects

### [01 · Do height and weight predict grip strength?](01-health-metrics-correlation/)
Correlation analysis with confidence intervals on a 10-observation health dataset.

The interesting result is a negative one: no predictor reaches significance, and every
95% interval spans zero. The notebook makes the case that the limiting factor is **sample
size, not effect size** — at *n* = 10 the interval on *r* is roughly ±0.65 regardless of
the data, so "we can't tell" is the only honest conclusion.

`pandas` · `scipy.stats` · `matplotlib`

<img src="01-health-metrics-correlation/results/grip_strength_scatter.png" width="620">

---

### [02 · What predicts student exam performance?](02-student-performance-eda/)
Exploratory analysis of 1,000 students across gender, parental education, lunch type, and
test preparation.

Test preparation shows the clearest separation — and it's the only variable in the dataset
a school can actually act on. Lunch type (a socioeconomic proxy) separates the
distributions nearly as sharply.

`pandas` · `matplotlib` · `seaborn`

> The source CSV isn't redistributed here — see [`data/README.md`](02-student-performance-eda/data/README.md)
> for a one-line download. Saved plots are preserved so the analysis reads without it.

---

### [03 · How well does a small sample represent its population?](03-diabetes-sampling-bootstrap/)
Sampling and bootstrap resampling against the 768-patient Pima Diabetes dataset, treated
as a known population.

A single 25-patient sample overstates mean glucose by **7.8%**. Bootstrapping — 500
resamples of 150 — recovers every blood-pressure statistic to within **0.9%**. The
notebook also shows that extremes (max) survive small samples well while central tendency
does not.

`pandas` · `numpy` · `seaborn`

<img src="03-diabetes-sampling-bootstrap/results/bootstrap_distributions.png" width="720">

---

### [04 · What were people tweeting about in early COVID-19?](04-covid-tweet-text-analysis/)
NLP on 3,798 tweets from March 2020: tokenisation, stopword removal, frequency analysis,
and a sentiment-split comparison.

Naive tokenisation makes `https` the most common token in the corpus — an artefact, not a
topic. After stripping URLs, mentions, and HTML entities, the real subject emerges:
**grocery supply**, not medicine. A log-ratio comparison then splits the vocabulary by
sentiment — `panic`, `fear`, `crisis`, `empty` on one side; `thank`, `support`, `safe`,
`small business` on the other.

`nltk` · `pandas` · `wordcloud` · `matplotlib`

<img src="04-covid-tweet-text-analysis/results/top_tokens.png" width="620">

---

### [05 · Segmenting credit-card customers with PCA and K-means](05-credit-card-customer-segmentation/)
Unsupervised clustering of 8,950 customers, scored by silhouette across raw, standardised,
and PCA-reduced feature spaces.

Raw features give the **highest** silhouette score — and the least useful clustering. Six
features are bounded in [0, 1] while `TENURE` runs 6–12, so that one feature dominates the
distance metric and K-means effectively clusters on tenure alone. Silhouette rewards the
resulting tight bands. The project's point is that **a higher score is not automatically a
better model**, and that silhouette is only comparable within the space the model was fitted in.

`scikit-learn` · `pandas` · `matplotlib`

> Dataset not redistributed — see [`data/README.md`](05-credit-card-customer-segmentation/data/README.md).
> Figures generate on first run.

---

## What these demonstrate

| | |
|---|---|
| **Statistical inference** | Pearson correlation, Fisher *z* confidence intervals, *p*-values, power reasoning |
| **Resampling** | Bootstrap sampling distributions; estimator bias and spread |
| **Data cleaning** | Categorical encoding, URL/entity stripping, missing-data handling |
| **NLP** | Tokenisation, stopword removal, frequency analysis, log-ratio class comparison |
| **Unsupervised ML** | K-means, PCA, silhouette scoring, elbow method, feature scaling |
| **Visualisation** | Distribution comparison, annotated multi-panel figures, word clouds |
| **Analytical judgment** | Reporting what the data *can't* support, and stating limitations |

## Setup

```bash
git clone https://github.com/m-nweke/data-analysis-portfolio.git
cd data-analysis-portfolio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Notebooks read data via paths relative to their own directory, so run them from within
each project's `notebooks/` folder.

## Repository layout

```
<project>/
├── README.md      question, method, findings
├── data/          inputs (raw/ and processed/ where applicable)
├── notebooks/     the analysis
└── results/       generated figures
```

## Notes

These analyses began as graduate coursework (CS5530 / CS5531) and have since been
reworked: bugs fixed, hardcoded Colab paths removed, methodology tightened, and findings
rewritten to state their own limitations. Project 05 additionally corrects two errors in
the original that invalidated its headline comparison — see its
[README](05-credit-card-customer-segmentation/) for what changed and why it mattered.

## License

[MIT](LICENSE)
