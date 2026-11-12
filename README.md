# Sephora Skincare: Exploratory Analysis and Machine Learning

Course project for the Machine Learning course: an analysis of 1,472 Sephora skincare products and three models that predict product properties from price, rating, brand and ingredient lists.

**Main result:** the ingredient list predicts whether a product suits oily skin (66% accuracy, 7 points above the baseline), while price, rating and brand do not (58%, below the 59% baseline).

## Questions

1. How are prices and ratings distributed, and which brands dominate the catalogue?
2. How well is each skin type covered, and which ingredients are most common?
3. Do expensive products get better ratings?
4. Can a model predict whether a product suits oily skin, and from which features?
5. How much of a product's price can be explained by its rating and brand?

## Data

`cosmetic_p.csv`: 1,472 skincare products with no missing values and no duplicate rows.

| Column | Description |
|---|---|
| `Label` | Product category, e.g. Moisturizer, Cleanser |
| `brand`, `name` | Brand and product name |
| `price` | Price in USD |
| `rank` | Average customer rating, 0–5 |
| `ingredients` | Comma-separated ingredient list |
| `Combination`, `Dry`, `Normal`, `Oily`, `Sensitive` | 1 if the product is labelled suitable for that skin type, else 0 |

## Approach

1. **Data quality:** data types, missing values, duplicates, descriptive statistics.
2. **Exploratory analysis:** price and rating distributions, top brands, ingredient frequency overall and by skin type.
3. **Patterns:** price vs rating, ratings across price tiers (cheap / medium / expensive), price vs number of supported skin types, brands by skin-type coverage.
4. **Models** (80/20 train/test split, `random_state=42`):
   - **Classification:** Random Forest predicting oily-skin suitability from price, rating and brand.
   - **Regression:** Random Forest predicting price from rating and brand.
   - **NLP classification:** TF-IDF on ingredient lists (1,000 features) with logistic regression predicting oily-skin suitability.

## Results

| Model | Target | Features | Result |
|---|---|---|---|
| Baseline (always "suitable") | Oily skin | none | 59% accuracy |
| Random Forest | Oily skin | price, rating, brand | 58% accuracy |
| TF-IDF + Logistic Regression | Oily skin | ingredient list | **66% accuracy**, recall 0.88 (suitable) / 0.36 (not suitable) |
| Random Forest Regressor | Price | rating, brand | R² = 0.35 |

### Key findings

- **Prices are right-skewed.** The median price is $42.50, while a tail of premium products pulls the mean up to $55.58 (maximum $370).
- **Ratings are high overall.** The median rating is 4.3 out of 5.
- **Sensitive skin is the least covered.** 65.6% of products suit combination skin, 65.2% normal, 61.4% dry, 60.7% oily and only 51.4% sensitive skin.
- **Price, rating and brand carry no signal for oily skin.** The Random Forest scores below the majority-class baseline, so these features are not useful for this task.
- **Ingredients do carry signal.** The TF-IDF model beats the baseline by 7 points, but it favours the majority class: it finds most suitable products and misses most unsuitable ones.
- **Brand and rating explain about a third of price variance.**
- **The base formula is shared.** Glycerin, phenoxyethanol and butylene glycol are among the most common ingredients for every skin type.

## Limitations and next steps

- Some products have a rating of 0, which most likely means "no reviews". They should be treated as missing values, not as a score.
- Ingredient names keep the leading space after splitting by comma, so the same ingredient can be counted twice. Stripping spaces would make frequencies exact.
- Brand is label-encoded as a number; one-hot or target encoding would suit linear models better.
- Class weights or resampling could raise recall for products that are not suitable for oily skin.
- Statistical tests (Spearman correlation for price vs rating, Kruskal–Wallis across price tiers) would confirm whether the visual patterns are significant.

## Repository structure

```
├── README.md
├── sephora_skincare_analysis.ipynb   # analysis and models
├── cosmetic_p.csv                    # dataset
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook sephora_skincare_analysis.ipynb
```

## Tech stack

Python, pandas, NumPy, Matplotlib, seaborn, scikit-learn (Random Forest, Logistic Regression, TF-IDF).

## Author

Anastasiya Kupreichyk · [github.com/anastasis3](https://github.com/anastasis3)
