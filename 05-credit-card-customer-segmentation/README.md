# Segmenting Credit-Card Customers with PCA and K-means

8,950 credit-card customers, no labels. Do they fall into distinct behavioural groups, and
how many?

**[→ Read the notebook](notebooks/pca_kmeans_segmentation.ipynb)**

## Data

*Credit Card Dataset for Clustering* — **not redistributed here.** See
[`data/README.md`](data/README.md) for a one-line download. The notebook runs end to end
once `CC.csv` is in place, generating every figure referenced below.

Seven behavioural features are used: balance frequency, purchase frequency (total, one-off,
instalment), cash-advance frequency, share of balance paid in full, and tenure.

## Method

K-means, evaluated with silhouette score across three feature representations — raw,
standardised, and PCA-reduced to two components. Cluster count chosen via the elbow method
and confirmed against silhouette. All seeds pinned (`random_state=42`).

## The central finding

**Raw features produce the highest silhouette score, and it is the least meaningful one.**

Six of the seven features are bounded in [0, 1]. `TENURE` runs 6 to 12. Because K-means
minimises squared Euclidean distance, that one wider-ranged feature dominates every
distance computation — so clustering unscaled data amounts to *clustering on tenure alone*.
Tenure is coarse and highly concentrated (most customers sit at 12 months), which yields
tight, well-separated bands. Silhouette rewards exactly that geometry, so the score looks
excellent while the segmentation recovers a variable already known and ignores the
purchasing behaviour of interest.

Standardising removes the dominance, lets all seven features contribute, and *lowers* the
score while making the clustering more informative.

**A higher silhouette is not automatically a better model.** That is the takeaway.

A second, subtler point the notebook is built around: a silhouette score is only comparable
**within the feature space the model was fitted in**. Fitting on scaled data and scoring
against the unscaled matrix — an easy and common slip — produces a number describing
neither model.

## Findings

1. Scaling is the decisive preprocessing step here, not PCA.
2. Silhouette must be computed in the same space the model was fitted in.
3. PCA to two components serves visualisation, not performance; the comparison table shows
   what the compression costs in retained variance.
4. The elbow is genuinely ambiguous between k = 3 and k = 4. Behavioural data rarely
   separates cleanly, and saying so beats picking the prettier plot.

## Limitations

- Silhouette measures geometric separation, not business usefulness. A well-separated
  segment is not necessarily an actionable one.
- K-means assumes roughly spherical, similarly sized clusters. Several of these features
  are skewed and zero-inflated, violating that assumption — DBSCAN or a Gaussian mixture
  would be a fairer comparison.
- Only 7 of 17 columns are used, inherited from the original analysis. The monetary columns
  (`BALANCE`, `PURCHASES`, `CREDIT_LIMIT`) are excluded and likely carry real signal.
- Median imputation assumes values are missing at random, which is untested.
- No ground-truth segmentation exists to validate against.

## Outputs

Generated into `results/` on first run: `elbow_scaled.png`,
`silhouette_comparison.png`, `pca_explained_variance.png`, `cluster_scatter_pca.png`.
