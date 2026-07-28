# What Predicts Student Exam Performance?

Exam scores for 1,000 students, alongside gender, race/ethnicity group, parental education,
lunch type, and whether the student completed a test-preparation course. Which of those
attributes actually track with performance?

**[→ Read the notebook](notebooks/student_performance_eda.ipynb)**

## Data

The *Students Performance in Exams* dataset. **Not redistributed here** — see
[`data/README.md`](data/README.md) for the download. The notebook's saved plots are
preserved, so the analysis is readable without fetching it.

## Method

1. Encode `test preparation course` (binary) and `race/ethnicity` (groups A–E) as integers.
2. Build a **composite score** — the unweighted mean of math, reading, and writing.
3. Compare the composite's distribution across each background attribute.

Collapsing three subjects into one composite is justified by the math/reading scatter in
the notebook: the subject scores move together tightly, so students are broadly strong or
broadly weak rather than specialised.

## Findings

1. **Test preparation shows the clearest separation.** Students who completed the course
   sit noticeably higher on the composite. It's also the only variable here a school can
   act on — the rest are background attributes.
2. **Lunch type separates the distributions.** Standard-lunch students outperform
   free/reduced-lunch students. Lunch is the dataset's socioeconomic proxy; read it that
   way, not as a claim about meals.
3. **Parental education trends upward** at each level, though adjacent categories overlap
   heavily enough that any single pair is unconvincing on its own.
4. **Composite scores are roughly normal**, centred near 68, with a left tail of low
   performers.

## Limitations

- Every comparison is **marginal** — no attribute is controlled for the others, so these
  effects are almost certainly entangled. Lunch type and parental education in particular
  are likely measuring overlapping things.
- **No significance testing.** Conclusions are read off distributions, so "noticeably
  higher" means visually separated, not statistically established.
- Group differences are purely descriptive. This dataset cannot say *why* a gap exists,
  and nothing here supports a causal reading.
- `race/ethnicity` is label-encoded 1–5, which implies an ordering that doesn't exist. This
  is safe here because the codes only ever split the data into groups — a model would
  require genuine one-hot encoding.

## Next step

Fit a regression with all attributes entered together, to see which retain explanatory
power once the others are accounted for.
