# Dataset

This project uses the **Credit Card Dataset for Clustering** (8,950 customers, 18 columns),
which is not redistributed in this repository.

## Download

Get it from Kaggle and save it here as `CC.csv`:

- https://www.kaggle.com/datasets/arjunbhasin2013/ccdata

```sh
kaggle datasets download -d arjunbhasin2013/ccdata -f "CC GENERAL.csv" --unzip
mv "CC GENERAL.csv" CC.csv
```

Final location:

```
05-credit-card-customer-segmentation/data/CC.csv
```

## Features used

The notebook uses seven behavioural columns. Note the scale difference — it's the point of
the analysis:

| Column | Range | Meaning |
|---|---|---|
| `BALANCE_FREQUENCY` | 0–1 | how often a balance is carried |
| `PURCHASES_FREQUENCY` | 0–1 | how often purchases are made |
| `ONEOFF_PURCHASES_FREQUENCY` | 0–1 | …as one-off purchases |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | 0–1 | …as instalment purchases |
| `CASH_ADVANCE_FREQUENCY` | 0–1 | how often cash advances are taken |
| `PRC_FULL_PAYMENT` | 0–1 | share of balance paid in full |
| `TENURE` | **6–12** | months as a customer |

`MINIMUM_PAYMENTS` (313 missing) and `CREDIT_LIMIT` (1 missing) are median-imputed.
