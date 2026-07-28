# Dataset

This project uses the **Students Performance in Exams** dataset (1,000 rows), which is not
redistributed in this repository.

## Download

Grab `StudentsPerformance.csv` from Kaggle and place it in this directory:

- https://www.kaggle.com/datasets/spscientist/students-performance-in-exams

```
03-student-performance-eda/data/StudentsPerformance.csv
```

The notebook reads it from `../data/StudentsPerformance.csv` and writes
`studentsPerformance_cleaned.csv` back here.

## Expected schema

| Column | Type | Values |
|---|---|---|
| `gender` | str | `male`, `female` |
| `race/ethnicity` | str | `group A` … `group E` |
| `parental level of education` | str | 6 levels, `some high school` → `master's degree` |
| `lunch` | str | `standard`, `free/reduced` |
| `test preparation course` | str | `none`, `completed` |
| `math score` | int | 0–100 |
| `reading score` | int | 0–100 |
| `writing score` | int | 0–100 |

The notebook's saved plot outputs are committed, so the analysis can be read in full
without downloading anything.
