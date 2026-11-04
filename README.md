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

## b. Dataset description

**Dataset name:** Online Shoppers Purchasing Intention Dataset
**Source:** UCI Machine Learning Repository (public dataset)
**Problem type:** Binary classification (Buy / Not buy)

### What the dataset contains

Each row is one visit (session) to an online shopping website. It records what the visitor did during the visit, such as which pages they viewed, how long they stayed, and when and how they arrived. The last column, `Revenue`, says whether the visit ended in a purchase. The data covers about one year of visits, from February to December.

| Item | Detail |
|---|---|
| Original rows | 12,330 visits |
| Rows used | 12,205 (125 repeated rows removed) |
| Features (clues) | 17 (10 numeric, 7 category-type) |
| Target column | `Revenue` (1 = bought, 0 = did not buy) |
| Missing values | None |
| Class balance | 15.6% buyers, 84.4% non-buyers (imbalanced) |

### Feature groups

| Group | Columns | Meaning |
|---|---|---|
| Page visits and time | `Administrative`, `Informational`, `ProductRelated` and their `_Duration` columns | Number of pages seen in each page type and the time spent on them |
| Website quality signals | `BounceRates`, `ExitRates`, `PageValues` | Share of visitors who leave after one page, how often a page is the last one seen, and the average value of pages seen before a purchase |
| Calendar | `Month`, `SpecialDay`, `Weekend` | Month of the visit, closeness to a special day such as Valentine's Day, and whether it was a weekend |
| Visitor details | `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType` | New or returning visitor, and codes for system, browser, region and traffic source |

### Why this dataset was chosen

- It meets the assignment limits: at least 12 features (it has 17) and at least 500 rows (it has 12,205).
- It is a real-world business problem from e-commerce.
- It mixes numeric and category columns, so it needs proper preprocessing (scaling and one-hot coding).
- It is imbalanced, so Accuracy alone is misleading, which makes the six metrics (Precision, Recall, F1 and MCC as well as Accuracy and AUC) meaningful.
- It is large enough that all five models (Logistic Regression, Decision Tree, KNN, Naive Bayes and SVM) can be compared fairly.

### Problem being solved

Most visitors to an online shop leave without buying. If the business can predict **while the visit is happening, or soon after, whether a visitor is likely to buy**, it can:

- show discounts or offers to visitors who are on the fence
- spend less advertising money on visitors unlikely to buy
- find out which behaviours (for example, high `PageValues` or low `ExitRates`) lead to a sale

The task is: given the details of a visit, predict `Revenue` (buy or not buy), then compare five models and pick the best one.

### What the exploration found

- Only 15.6% of visits end in a purchase, so a model that always says "no buy" would already be about 84% accurate. This is why MCC and F1 are used to judge the models.
- 77.6% of visits have `PageValues` = 0, and only 3.9% of those buy. When `PageValues` is above 0, 56.3% buy. It is the strongest clue.
- November has the highest buying rate (about 25%).
- New visitors buy more often (about 25%) than returning visitors (about 14%).

### Data preparation

Repeated rows were removed. True/False columns were changed to 1/0. Number columns were scaled with StandardScaler. `Month` and `VisitorType` were converted to 0/1 columns with One-Hot encoding. The data was split 80% training (9,764 rows) and 20% test (2,441 rows, saved as `test_data.csv`) with a fixed random state and the same buyer ratio in both parts.

---

## c. GitHub repository link

**GitHub:** https://github.com/SudharsanK0221/ML_Assignment_2

**Live Streamlit app:** https://mlassignment2-tn526quxcx3tb4yimvfpsx.streamlit.app/

<img width="900" height="466" alt="image" src="https://github.com/user-attachments/assets/53649fe9-c484-4d6c-9069-20a6c3f5df78" />

<img width="900" height="484" alt="image" src="https://github.com/user-attachments/assets/4ef65a97-55eb-4ba5-aee3-781d915d43f2" />

<img width="900" height="440" alt="image" src="https://github.com/user-attachments/assets/e53e1fbd-6f1c-4ac2-8d9d-0d0db98e986b" />

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

### Model Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8894 | 0.9022 | 0.7718 | 0.4162 | 0.5408 | 0.5142 |
| Decision Tree | 0.8980 | 0.9122 | 0.6997 | 0.6099 | 0.6517 | 0.5942 |
| KNN | 0.8800 | 0.8208 | 0.7051 | 0.4005 | 0.5109 | 0.4716 |
| Naive Bayes | 0.7108 | 0.8268 | 0.3284 | 0.8115 | 0.4676 | 0.3757 |
| Support Vector Machine | 0.8943 | 0.8799 | 0.7403 | 0.5000 | 0.5969 | 0.5524 |

Settings used: Logistic Regression (max_iter = 1000), Decision Tree (max_depth = 6), KNN (k = 7), Gaussian Naive Bayes (default), SVM (RBF kernel, probability = True). Precision, Recall and F1 are for the "buyer" class (Revenue = 1).

### Confusion matrix of each model (test data)

Test data = 2,441 visits: 382 real buyers and 2,059 real non-buyers. TN = correct "no buy", FP = false alarm, FN = missed buyer, TP = correct "buy".

| ML Model Name | TN | FP | FN | TP | Buyers found |
|---|---|---|---|---|---|
| Logistic Regression | 2012 | 47 | 223 | 159 | 159 of 382 (41.6%) |
| Decision Tree | 1959 | 100 | 149 | 233 | 233 of 382 (61.0%) |
| KNN | 1995 | 64 | 229 | 153 | 153 of 382 (40.1%) |
| Naive Bayes | 1425 | 634 | 72 | 310 | 310 of 382 (81.2%) |
| Support Vector Machine | 1992 | 67 | 191 | 191 | 191 of 382 (50.0%) |

### What each metric tells us on this dataset

| Metric | What it shows here | Best model | Weakest model |
|---|---|---|---|
| Accuracy | Share of all 2,441 guesses that were right. Always saying "no buy" already gives 84.4%, so a model must beat this to be useful. Only Naive Bayes fails (71.1%). | Decision Tree (0.8980) | Naive Bayes (0.7108) |
| AUC | Chance that a real buyer gets a higher buy-score than a real non-buyer. Does not depend on the cut-off. | Decision Tree (0.9122) | KNN (0.8208) |
| Precision | Of the visits flagged "will buy", how many really bought. Low precision means wasted offers. | Logistic Regression (0.7718) | Naive Bayes (0.3284) |
| Recall | Of the 382 real buyers, how many were found. Low recall means lost sales. | Naive Bayes (0.8115) | KNN (0.4005) |
| F1 | Balance of Precision and Recall in one number. Punishes a model that is good at only one of them. | Decision Tree (0.6517) | Naive Bayes (0.4676) |
| MCC | Uses all four boxes of the confusion matrix. The fairest single score for imbalanced data (1 = perfect, 0 = guessing). | Decision Tree (0.5942) | Naive Bayes (0.3757) |

### Observations about each model

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Good accuracy (88.9%) and a high AUC (0.90), so it separates buyers from non-buyers well. It is careful: when it says "will buy" it is right about 77% of the time (best precision). But it misses many real buyers (recall only 0.42), so F1 and MCC are only average. It is simple, fast and easy to explain. |
| Decision Tree | The best model on this data. It has the highest accuracy (89.8%), AUC (0.91), F1 (0.65) and MCC (0.59). It catches about 61% of real buyers while keeping precision near 70%. The depth was limited to 6, which stops it from just memorising the training data. It probably works well because a few clues (like PageValues and ExitRates) decide most of the buying behaviour and a tree can pick these directly. |
| KNN | Accuracy (88.0%) looks fine, but this is mostly because most visitors do not buy. It has the lowest AUC (0.82) and the lowest recall (0.40), so it misses most buyers. It guesses by looking at the nearest similar visits, and with many columns (after One-Hot) the "nearest" visits are not always truly similar. It is also slower at prediction time and gives the weakest MCC after Naive Bayes. |
| Naive Bayes | The weakest in accuracy (71.1%) and precision (0.33), but it has the **highest recall (0.81)**: it catches most buyers, though it also wrongly marks many non-buyers as buyers. The reason is that it assumes all clues are independent of each other, which is not true here (for example pages seen and time spent move together). It is useful only if missing a buyer is much worse than a false alarm. |
| Support Vector Machine | Second best overall. Accuracy 89.4%, AUC 0.88, precision 0.74 and MCC 0.55. It balances precision and recall better than Logistic Regression and KNN, but it still misses half of the buyers (recall 0.50). It also took the longest to train, and the Decision Tree still beat it on every important score. |
| **Overall, Winner for your dataset?** | **Decision Tree.** It has the best Accuracy, AUC, F1 and MCC. MCC is the most reliable score for this imbalanced data (only ~15% buyers), and the Decision Tree is also the fastest to train and the easiest to explain. SVM is the runner-up. |

## Streamlit App Features Implemented
As per the assignment requirements, the deployed Streamlit application includes:
- **Dataset upload option:** Users can upload the `test_data.csv` to evaluate models without retraining.
<img width="275" height="270" alt="image" src="https://github.com/user-attachments/assets/ef20ca42-9f1e-4ab1-9ce9-a930201b5552" />

- **Model selection dropdown:** Users can switch between Logistic Regression, Decision Tree, KNN, Naive Bayes, and SVM.
<img width="317" height="314" alt="image" src="https://github.com/user-attachments/assets/639a14db-1dd0-4c1b-9786-9acedf132330" />

- **Display of evaluation metrics:** The app dynamically calculates and displays Accuracy, AUC, Precision, Recall, F1, and MCC on the uploaded test data.
<img width="1883" height="403" alt="image" src="https://github.com/user-attachments/assets/f8561fa1-3864-4aea-949f-ba79c4fa14e0" />

- **Visualizations:** The app outputs the Confusion Matrix and a detailed Classification Report for the selected model.
   
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
