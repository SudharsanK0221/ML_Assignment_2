# Online Shoppers Buying Prediction

**Machine Learning - Assignment 2 | M.Tech (AIML), BITS Pilani WILP**
Name: SUDHARSAN K | BITS ID: 2026ak05405

---

## a. Problem statement

A shopping website gets many visitors, but only a small number of them buy something. If the website can guess **early** which visits will end in a purchase, it can show offers to the right people and save marketing cost.

This project builds **5 classification models** that look at the details of one website visit (pages seen, time spent, bounce and exit rates, month, type of visitor, etc.) and guess whether the visit ends **with a purchase (Revenue = 1)** or **without a purchase (Revenue = 0)**. The models are compared using 6 scores and shown in an interactive Streamlit web app.

---

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository |
| Problem type | Binary classification |
| Original rows | 12,330 (one row = one visit / session) |
| Rows used | 12,205 (125 repeated rows were removed) |
| Clue columns (features) | 17 |
| Answer column (target) | `Revenue` (1 = bought something, 0 = did not buy) |
| Missing values | None |
| Class balance | About 15.6% buyers and 84.4% non-buyers (imbalanced data) |

**Meaning of the columns**

| Column | Simple meaning |
|---|---|
| Administrative, Administrative_Duration | Number of account/admin pages visited and time spent on them |
| Informational, Informational_Duration | Number of information pages visited and time spent on them |
| ProductRelated, ProductRelated_Duration | Number of product pages visited and time spent on them |
| BounceRates | Share of visits that left after seeing only one page |
| ExitRates | Share of views of a page that were the last page of the visit |
| PageValues | Average value of the pages seen before a purchase |
| SpecialDay | How close the visit date is to a special day (like Valentine's Day) |
| Month | Month of the visit |
| OperatingSystems, Browser, Region, TrafficType | Codes (numbers) for the visitor's system, browser, region and how they reached the site |
| VisitorType | New visitor, returning visitor or other |
| Weekend | Whether the visit was on a weekend |
| Revenue | **Answer column**: did the visit end in a purchase |

**Data preparation:** repeated rows removed, True/False columns changed to 1/0, number columns scaled (StandardScaler), word columns (`Month`, `VisitorType`) changed to 0/1 columns (One-Hot). The data was split 80% training / 20% test (random_state = 42, stratified). All steps are saved inside each model's pipeline, so the app does the same cleaning automatically.

---

## c. GitHub repository link

**GitHub:** https://github.com/SudharsanK0221/ML_Assignment_2

**Live Streamlit app:** https://mlassignment2-tn526quxcx3tb4yimvfpsx.streamlit.app/

```
project-folder/
│-- app.py                                  (Streamlit app)
│-- shopper_buying_prediction.ipynb         (notebook: data check, training, scores, saving models)
│-- requirements.txt
│-- README.md
│-- test_data.csv                           (20% test data with all 17 columns + Revenue)
│-- online_shoppers_intention.csv           (original dataset)
│-- model/
      │-- logistic_regression.pkl
      │-- decision_tree.pkl
      │-- knn.pkl
      │-- naive_bayes.pkl
      │-- svm.pkl
```

**How to run on your own computer**
```
pip install -r requirements.txt
streamlit run app.py
```
To re-create the models, open `shopper_buying_prediction.ipynb` and run all cells from top to bottom.

---

## d. Models used

All 5 models were trained on the **same training data** and scored on the **same test data** (2,441 rows).

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8894 | 0.9022 | 0.7718 | 0.4162 | 0.5408 | 0.5142 |
| Decision Tree | 0.8980 | 0.9122 | 0.6997 | 0.6099 | 0.6517 | 0.5942 |
| KNN | 0.8800 | 0.8208 | 0.7051 | 0.4005 | 0.5109 | 0.4716 |
| Naive Bayes | 0.7108 | 0.8268 | 0.3284 | 0.8115 | 0.4676 | 0.3757 |
| Support Vector Machine | 0.8943 | 0.8799 | 0.7403 | 0.5000 | 0.5969 | 0.5524 |

Settings used: Logistic Regression (max_iter = 1000), Decision Tree (max_depth = 6), KNN (k = 7), Gaussian Naive Bayes (default), SVM (RBF kernel, probability = True). Precision, Recall and F1 are for the "buyer" class (Revenue = 1).

### Observations about each model

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Good accuracy (88.9%) and a high AUC (0.90), so it separates buyers from non-buyers well. It is careful: when it says "will buy" it is right about 77% of the time (best precision). But it misses many real buyers (recall only 0.42), so F1 and MCC are only average. It is simple, fast and easy to explain. |
| Decision Tree | The best model on this data. It has the highest accuracy (89.8%), AUC (0.91), F1 (0.65) and MCC (0.59). It catches about 61% of real buyers while keeping precision near 70%. The depth was limited to 6, which stops it from just memorising the training data. It probably works well because a few clues (like PageValues and ExitRates) decide most of the buying behaviour and a tree can pick these directly. |
| KNN | Accuracy (88.0%) looks fine, but this is mostly because most visitors do not buy. It has the lowest AUC (0.82) and the lowest recall (0.40), so it misses most buyers. It guesses by looking at the nearest similar visits, and with many columns (after One-Hot) the "nearest" visits are not always truly similar. It is also slower at prediction time and gives the weakest MCC after Naive Bayes. |
| Naive Bayes | The weakest in accuracy (71.1%) and precision (0.33), but it has the **highest recall (0.81)**: it catches most buyers, though it also wrongly marks many non-buyers as buyers. The reason is that it assumes all clues are independent of each other, which is not true here (for example pages seen and time spent move together). It is useful only if missing a buyer is much worse than a false alarm. |
| Support Vector Machine | Second best overall. Accuracy 89.4%, AUC 0.88, precision 0.74 and MCC 0.55. It balances precision and recall better than Logistic Regression and KNN, but it still misses half of the buyers (recall 0.50). It also took the longest to train, and the Decision Tree still beat it on every important score. |
| **Overall, Winner for your dataset?** | **Decision Tree.** It has the best Accuracy, AUC, F1 and MCC. MCC is the most reliable score for this imbalanced data (only ~15% buyers), and the Decision Tree is also the fastest to train and the easiest to explain. SVM is the runner-up. |

**Note on imbalanced data:** since only about 15% of visits end in a purchase, a model that always says "no buy" would already get about 84% accuracy. That is why Precision, Recall, F1 and MCC are looked at together with Accuracy.

---

## Streamlit app features

- Upload a test CSV (or use the sample `test_data.csv` with one click)
- Dropdown to choose any of the 5 saved models
- All 6 evaluation scores shown for the chosen model
- Confusion matrix and classification report (plus ROC curve)
- A tab that compares all 5 models on the uploaded data
- The app only **loads** the saved models. It never re-trains.

## BITS Virtual Lab
<img width="1661" height="812" alt="image" src="https://github.com/user-attachments/assets/de9c549b-f6ed-406a-96d3-fcab2d053c3d" />


## Links
**GitHub:** https://github.com/SudharsanK0221/ML_Assignment_2

**Live Streamlit app:** https://mlassignment2-tn526quxcx3tb4yimvfpsx.streamlit.app/
