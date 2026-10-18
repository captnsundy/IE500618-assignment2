# Assignment 2: Diabetes Classification — Written Report

IE500618 — Group 13: Rune-André Gamlem, Magnus Grande, Ida Soldal

Word Count: ~1880 including questions (~1650 answers only)

## Part A - Answers to Q1-Q7

***Q1. Understanding the data: What do class 0 and class 1 mean? Is the target balanced? Why does the class distribution matter when evaluating the model?***

**Answer:**

Class 0 means no diabetes, and class 1 means prediabetes **or** diabetes. The two are merged, so the model cannot tell prediabetes and diabetes apart. We also checked that the target only contains 0 and 1, since the guide mentions an older three-level coding (0/1/2) that indeed does not apply to this file.

The target is not balanced: `218,334` respondents (86.1%) are class 0 and `35,346` (13.9%) are class 1.

This matters because a "model" that always predicts class 0 is already right 86.1% of the time without learning anything. Accuracy will therefore look high almost regardless of the model, so we compare against this baseline and focus on precision and recall for class 1. It is also why the split is stratified (Q4).

The imbalance affects the model as well, not only the evaluation. Logistic Regression's predicted probabilities average out to the share of class 1 in TRAIN (13.9%), so most respondents get a low probability and few reach the 0.5 needed to be predicted as class 1: only 4.0% of TEST is predicted as class 1, against 13.9% who actually are. This is a large part of why recall is low (Q5) and why lowering the threshold helps (Q6).

***Q2. Understanding the features: Choose three features of different types. What do their values mean? Why is looking only at the numbers stored in the CSV not enough to understand a feature?***

**Answer:**

- `BMI` (numerical measurement, 12-98): a measured body mass index. `BMI = 30` is a BMI of 30, and the distances between values are real.
- `HighBP` (binary categorical, 0/1): `1` means the respondent has high blood pressure, `0` means no. The 1 is a label, not an amount, and its mean (0.429) is the share answering yes.
- `Age` (ordinal categorical, 1-13): an age **group**. `Age = 9` means 60-64 years old, not 9 years. The groups are ordered but not equally wide (most cover 5 years, 18-24 covers 7, and 80+ is open-ended).

All 22 columns are stored as `float64`, but only `BMI`, `MentHlth` and `PhysHlth` are numbers in the usual sense. The rest are categories stored as numbers. Without the guide we would read `Age = 9` as a nine-year-old, treat the codes as measured distances, and get the direction wrong for some features: for `GenHlth` a higher code means **worse** health (1 = excellent, 5 = poor), while for `Income` a higher code means higher income. The meaning decides both how to explore a feature and how to preprocess it (see Step 1 and Step 4 of the notebook).

***Q3. EDA: You explored all input features. Which three to five findings were most informative or interesting? What did you observe? What can you conclude from these results, and what can you not conclude?***

**Answer:**

We compared proportions of class 1 rather than counts, since class 1 is about six times smaller (Step 2 of the notebook).

1. **Self-rated general health** (`GenHlth`) had the strongest relationship: class 1 rises at every step, from `2.5%` for "excellent" to `37.9%` for "poor".
2. **High blood pressure and high cholesterol** are both common (about 43% answer yes) and go with much more class 1 (`HighBP`: 24.4% vs 6.0%; `HighChol`: 22.0% vs 8.0%).
3. **BMI** is shifted upwards for class 1 (median 31 vs 27), but the classes overlap heavily. The values above 70 behave oddly: class 1 rises steadily with BMI up to `39.4%` for BMI 50-70, then drops to `13.0%` above 70, and those values cluster on a few specific numbers. We could not tell which values are wrong, so we kept them (0.23% of the rows) and listed it as a limitation.
4. **Age** rises steadily from `1.4%` (18-24) to `21.8%` (70-74), then levels off and drops to `18.5%` for 80+, so it is not one straight-line relationship.
5. **Surprising directions:** heavy drinkers have *less* class 1 (5.8% vs 14.4%), and respondents without a recent cholesterol check have very little (2.5% vs 14.4%).

We also found that `GenHlth`, `PhysHlth` and `DiffWalk` overlap a lot. `Fruits`, `Smoker`, `Sex`, `AnyHealthcare`, `NoDocbcCost` and `MentHlth` showed little relationship with the target on their own.

Conclusion: these features are associated with class 1 in this survey. Can't conclude: that any of them cause diabetes. Finding 5 shows why: heavy drinking is not protective, and a cholesterol check does not cause diabetes. More likely, a check means contact with health care (where diabetes gets diagnosed), and people diagnosed with diabetes may have been told to drink less. The survey is a single snapshot, so it cannot show what came first.

***Q4. TRAIN and TEST: Why must TEST remain separate while preprocessing and the model are fitted?***

**Answer:**

We used an 80/20 split with `stratify=y` (`random_state=69`), giving 202,944 TRAIN and 50,736 TEST rows, both with 13.93% class 1. TEST was not balanced or resampled.

TEST stands in for respondents the model has never seen. If anything is learned from it, such as the scaler's means and standard deviations, or the choice of threshold, information from TEST leaks into the model and the TEST score becomes too optimistic.

***Q5. Classification errors and metrics: Using your confusion matrix, explain false positives and false negatives for this diabetes problem. What do accuracy, precision and recall tell you? Why is accuracy alone not enough?***

**Answer:**

Confusion matrix on TEST (threshold 0.50): TN = `42,791`, FP = `876`, FN = `5,932`, TP = `1,137`.

- A **false positive** is a respondent without diabetes predicted as prediabetes/diabetes: an unnecessary worry and a follow-up test.
- A **false negative** is a respondent with prediabetes or diabetes predicted as "no diabetes": a missed case. Since the point is early identification, this is arguably the more serious error.

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Always predict 0 (baseline) | 0.861 | 0.000 | 0.000 | 0.000 |
| Logistic Regression, TEST | 0.866 | 0.565 | 0.161 | 0.250 |
| Logistic Regression, TRAIN | 0.863 | 0.531 | 0.151 | 0.236 |

Accuracy is the share of all respondents classified correctly. Precision is the share of those flagged as class 1 who really are class 1 (56.5%). Recall is the share of actual class 1 respondents that the model found: only 16.1%, so 84% are missed.

Accuracy alone is not enough because the model's 0.866 is only 0.5 percentage points above always predicting "no diabetes", and that baseline finds no one at all. Accuracy hides the low recall completely. TRAIN and TEST are almost identical, so the model is not overfitting. The predicted probabilities of the two classes simply overlap a lot, and the missed class 1 respondents look much less extreme in their answers than the ones that were found. With these features and a linear model, many class 1 respondents cannot be told apart from class 0. A more flexible model might separate some of them better, but we did not test that.

***Q6. Probability and threshold: Compare thresholds 0.50 and 0.40. What happened to false positives, false negatives, precision and recall? What did not change when you changed only the threshold?***

**Answer:**

| Threshold | Accuracy | Precision | Recall | F1    | False positives | False negatives |
|-----------|----------|-----------|--------|-------|-----------------|-----------------|
| 0.50      | 0.866    | 0.565     | 0.161  | 0.250 | 876             | 5,932           |
| 0.40      | 0.860    | 0.497     | 0.276  | 0.355 | 1,972           | 5,120           |

Lowering the threshold means a lower probability is enough to predict class 1, so more respondents are flagged (2,013 → 3,921). False negatives went down by 812 and false positives went up by 1,096, about 1.35 false alarms per extra case found. Recall rose (0.161 → 0.276) and precision fell (0.565 → 0.497). Accuracy fell slightly and is now just below the always-0 baseline, even though the model finds more cases.

What did not change: the fitted model (coefficients and preprocessing), the predicted probabilities, and the TEST data with its actual classes. Both confusion matrices have 43,667 actual class 0 and 7,069 actual class 1 respondents. Only the step from probability to class changed. Which threshold is better depends on how costly a missed case is compared to a false alarm, and if we were to choose one, it should be chosen without using TEST.

***Q7. Conclusion: What did you learn from the data and the Logistic Regression model? Give at least two limitations of this analysis.***

**Answer:**

From the data: class 1 is more common with worse self-rated health, high blood pressure, high cholesterol, heart disease, difficulty walking, higher BMI and age, and lower income and education. From the model: Logistic Regression picks up mostly the same patterns, but at threshold 0.50 it finds only 16% of class 1 and barely beats the baseline on accuracy. Lowering the threshold improves recall at the cost of precision without changing the model. We also learned to check the coefficients against the EDA: `PhysHlth` gets a negative coefficient because `GenHlth` and `DiffWalk` already carry the same information.

Limitations:

1. **Self-reported, single-snapshot survey.** We cannot tell what came first, and some associations (drinking, cholesterol checks, bad health days) may be consequences of a diagnosis rather than risk factors. Nothing shows causation.
2. **Class 1 merges prediabetes and diabetes**, so the model cannot separate them.
3. **Ordinal features as one straight-line effect.** One coefficient per ordinal feature assumes equal steps between categories, which does not fit the levelling-off of `Age`.
4. **Data quality.** The BMI values above 70 look suspicious and the day counts are rounded by respondents. We kept both, since we could not tell which values are wrong.
5. **One random split, and thresholds not chosen from costs.**

## Part B - Group-work reflection

***G1. How was the work divided among group members, and what did each member contribute?***

**Answer:**

We started with a short discussion of our ambitions for the assignment and what we had learned from assignment 1, and then split the work into three parts:

- **Ida Soldal** set up the project structure: the repository, the notebook skeleton following the eight steps of the assignment, and the report template with the questions.
- **Magnus Grande** wrote the initial draft on top of that skeleton: the EDA, the preprocessing pipeline, the model, the evaluation and threshold comparison, and the answers to Q1-Q7.
- **Rune-André Gamlem** reviewed the technical implementation: whether the pipeline kept TEST separate, whether the numbers in the report matched the notebook, and whether the interpretations held up. This led to clarifications in Q1 and Q5.

Due to time constraints we were unable to cowork like we had intended, so we instead opted to split the work and dedicate more time to learning the material individually. All changes were reviewed by the group before submission.

***G2. How did your group make sure that everyone understood the complete solution, not only the part they worked on?***

**Answer:**

Compared to assignment 1, this assignment had a much smaller and more clearly defined scope. The steps were given as instructions to follow, there was no feature engineering, and no rows or features were removed, so there were few places where the solution could go in different directions. Because the skeleton followed the assignment's steps exactly, each member could read the finished notebook step by step against the instructions and see how every requirement was solved.

Before submission, every member read through the final notebook and report, so that each of us understands the complete technical solution and can explain how it differs from assignment 1.
