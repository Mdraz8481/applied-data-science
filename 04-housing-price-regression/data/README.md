# Dataset

This project uses the **Ames, Iowa housing dataset** — 1,460 residential sales from
2006–2010, 79 descriptive features plus `SalePrice`. It is not redistributed here.

## Download

Get the training file from the Kaggle competition and save it in this directory as
`houses_dataset.csv`:

- https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

```sh
kaggle competitions download -c house-prices-advanced-regression-techniques -f train.csv
mv train.csv houses_dataset.csv
```

Final location:

```
04-housing-price-regression/data/houses_dataset.csv
```

The dataset originates with Dean De Cock, *Ames, Iowa: Alternative to the Boston Housing
Data as an End of Semester Regression Project*, Journal of Statistics Education 19(3),
2011.

## Columns used

The notebook uses only the numeric columns — 36 features after dropping `Id` and the
target. The 43 categorical columns, including `Neighborhood`, are deliberately excluded;
see the project README's limitations for why that matters.

| Column | Meaning | Correlation with `SalePrice` |
|---|---|---|
| `OverallQual` | assessor's 1–10 quality rating | 0.79 |
| `GrLivArea` | above-grade living area, sq ft | 0.71 |
| `GarageCars` | garage capacity, cars | 0.64 |
| `GarageArea` | garage size, sq ft | 0.62 |
| `TotalBsmtSF` | basement area, sq ft | 0.61 |

Three numeric columns have missing values — `LotFrontage` (259), `GarageYrBlt` (81), and
`MasVnrArea` (8). They are median-imputed inside the modelling pipeline so the imputation
is refit within each cross-validation fold rather than on the full dataset.
