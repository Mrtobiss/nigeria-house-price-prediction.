# Results: Nigeria House Price Prediction

**Dataset source:** [Add the dataset source link here]
**Licence / terms of use:** [Add the licence or terms of use here]

### Result
- **Selected model:** Gradient Boosting, chosen by 5-fold cross-validation on 10,853 training listings.
- **Hold-out performance** (2,714 unseen listings, random split): R2 = 0.76 on log price, MAE = 0.40 and RMSE = 0.54 on the log scale, median error 31%, and 49% of predictions within 30% of the actual asking price.
- **Against simple rules:** the grouped median (town + property type) reaches R2 = 0.69 and a median error of 35%; the global median reaches R2 = -0.00.
- **Ceiling:** listings with identical inputs differ in price by about x1.67 (one SD), so without property size no model can reach much beyond R2 of roughly 0.79.
- **Error by region** (median %): Lagos 30%; Abuja 29%; Other 34%.
- **Price range covered:** ₦1.5M to ₦7.5B. 197 records priced more than 10 times above or below comparable listings were removed; the scores apply to the retained range.

### Data preparation in one line
24,326 raw rows -> 13,567 modelling rows: 1 unresolvable state, 10,561 exact duplicates, 197 implausible prices removed ("Anambara" labels corrected first; price references computed from training data only).

### Assumptions
- Prices are asking prices from listings, not confirmed sale prices.
- Exact repeated listings were treated as duplicates (no listing ID, size or date exists to prove otherwise).
- "Anambara" state labels were treated as errors and reassigned from internal evidence and general geography (Task 15), not from an external reference.
- The test set is a random split of de-duplicated listings, so scores describe performance on listings *similar to ones the model has seen*.

### Limitations
- No property size, age, condition or exact address: identical-looking listings differ widely in price.
- Heavy concentration in Lagos and Abuja; other states have few listings and larger errors.
- Towns with fewer than 5 training listings have a median error of 55%, against 30% for well-covered towns.
- The model was not validated on towns it has never seen, so the app offers only towns present in the training data.
- Scores apply only to the retained price range; extreme prices were removed by the rule in Task 18.
- The model predicts the typical asking price for listings like the one described, not the price a buyer would pay or a valuation.
- Prices change over time and the data has no date.
- The dataset source and licence are placeholders until completed by the author.

### Next steps
Deploy `app.py` on Streamlit Community Cloud, complete the executive summary, pitch deck and video from the figures in `reports/figures/`, and (if possible) collect property size to raise the accuracy ceiling.

Notebook run time: 2.3 minutes.