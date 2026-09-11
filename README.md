# Outlier Handling in Machine Learning

Outlier handling is an important part of **data preprocessing** in Machine Learning. In this project, we identify outliers using the **IQR (Interquartile Range) method** and handle them using two techniques:

1. **Trimming**
2. **Capping (Winsorization)**

---

## 1. What is an Outlier?

An **outlier** is a data point that is significantly different from most of the other observations in a dataset.

For example:

```text
10, 12, 11, 13, 12, 14, 100
```

Here, `100` is an outlier because most of the values are between `10` and `14`.

Outliers can occur because of:

* Data entry mistakes
* Measurement errors
* Sensor errors
* Incorrect data collection
* Genuine but unusual observations
* Rare events

An outlier is **not always an error**. Sometimes it represents a real and important observation.

---

# 2. Why Do We Handle Outliers?

Outliers can have a significant effect on the data and on some Machine Learning algorithms.

For example:

```text
10, 12, 11, 13, 12, 100
```

The value `100` can increase the mean significantly compared with the majority of the values.

Outliers can affect:

* Mean
* Variance
* Standard deviation
* Correlation
* Data distribution
* Model performance

Some Machine Learning algorithms are more sensitive to outliers than others.

### Algorithms that can be affected by outliers

* Linear Regression
* Logistic Regression
* K-Nearest Neighbors (KNN)
* K-Means Clustering
* Support Vector Machines (SVM)
* Neural Networks

Tree-based algorithms such as Decision Trees and Random Forests are generally less sensitive to outliers.

Therefore, before handling an outlier, we should understand whether it is:

> **A genuine observation or an incorrect/extreme value.**

---

# 3. Dataset

In this project, we are working with:

```python
newplacementdata.csv
```

We load the dataset using Pandas:

```python
import numpy as np
import pandas as pd

df = pd.read_csv('newplacementdata.csv')

df.head(3)
```

The column we are interested in is:

```text
placement_exam_marks
```

---

# 4. Understanding the Data

We first use the `describe()` function to understand the statistical properties of the column.

```python
df['placement_exam_marks'].describe()
```

This gives information such as:

* Count
* Mean
* Standard deviation
* Minimum
* 25th percentile
* 50th percentile
* 75th percentile
* Maximum

Example:

```text
count
mean
std
min
25%
50%
75%
max
```

This helps us understand the distribution of the data.

---

# 5. Detecting Outliers Using a Box Plot

A box plot is one of the easiest ways to visually identify outliers.

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(x=df['placement_exam_marks'])
```

A box plot contains:

* Minimum value
* Q1 (25th percentile)
* Median (50th percentile)
* Q3 (75th percentile)
* Maximum value
* Potential outliers

Values appearing beyond the whiskers are generally considered potential outliers.

---

# 6. IQR Method

One of the most commonly used methods for detecting outliers is the **IQR method**.

IQR stands for:

> **Interquartile Range**

The IQR represents the range containing the **middle 50% of the data**.

To calculate IQR, we need:

* Q1 → 25th percentile
* Q3 → 75th percentile

The formula is:

```text
IQR = Q3 - Q1
```

---

# 7. Finding Q1 and Q3

We calculate the 25th and 75th percentiles using Pandas:

```python
percentile25 = df['placement_exam_marks'].quantile(0.25)

percentile75 = df['placement_exam_marks'].quantile(0.75)

print("percentile25 is :", percentile25)
print("percentile75 is :", percentile75)
```

Here:

```text
Q1 = 25th percentile
Q3 = 75th percentile
```

---

# 8. Calculating IQR

The formula is:

```text
IQR = Q3 - Q1
```

In Python:

```python
IQR = percentile75 - percentile25

IQR
```

### Example

Suppose:

```text
Q1 = 20
Q3 = 40
```

Then:

```text
IQR = 40 - 20
    = 20
```

Therefore, the middle 50% of the data lies within an interval of `20`.

---

# 9. Finding Upper and Lower Limits

To identify potential outliers, we calculate the lower and upper limits.

### Lower Limit

```text
Lower Limit = Q1 - 1.5 × IQR
```

### Upper Limit

```text
Upper Limit = Q3 + 1.5 × IQR
```

In Python:

```python
upper_limit = percentile75 + 1.5 * IQR

lower_limit = percentile25 - 1.5 * IQR

print("Upper Limit:", upper_limit)
print("Lower Limit:", lower_limit)
```

The value `1.5` is a commonly used rule in the IQR method.

---

# 10. Identifying Outliers

### Upper Outliers

We can find observations greater than the upper limit:

```python
df[df['placement_exam_marks'] > upper_limit]
```

These are potential upper outliers.

### Lower Outliers

We can find observations below the lower limit:

```python
df[df['placement_exam_marks'] < lower_limit]
```

These are potential lower outliers.

Therefore:

```text
Value > Upper Limit
        ↓
Upper Outlier


Value < Lower Limit
        ↓
Lower Outlier
```

---

# 11. Two Techniques for Handling Outliers

There are two techniques used in this project:

```text
Outlier Handling
       |
       |----------------------|
       ↓                      ↓
   Trimming                 Capping
```

---

# 12. Technique 1 — Trimming

## What is Trimming?

**Trimming** means completely removing the observations that contain outliers.

For example:

```text
10, 12, 11, 13, 12, 100
```

If `100` is identified as an outlier, trimming removes the observation containing `100`.

The resulting data becomes:

```text
10, 12, 11, 13, 12
```

### Advantages

* Simple to implement
* Removes extreme observations completely
* Useful when outliers are clearly incorrect

### Disadvantages

* We lose data
* Dataset size decreases
* Genuine observations may be removed
* If there are many outliers, too much information can be lost

---

# 13. Trimming in This Project

We create a new DataFrame:

```python
newdf = df[df['placement_exam_marks'] < upper_limit]
```

This keeps only the observations whose:

```text
placement_exam_marks < upper_limit
```

Therefore, values greater than or equal to the upper limit are removed.

### Important Note

This particular code handles **only upper outliers**.

It does not remove lower outliers.

If we wanted to remove both upper and lower outliers, we could use:

```python
newdf = df[
    (df['placement_exam_marks'] >= lower_limit) &
    (df['placement_exam_marks'] <= upper_limit)
]
```

This removes observations outside both limits.

---

# 14. Visualizing Trimming

We compare the original data with the trimmed data:

```python
plt.figure(figsize=(15, 5))

plt.subplot(121)

sns.boxplot(x=df['placement_exam_marks'])

plt.subplot(122)

sns.boxplot(x=newdf['placement_exam_marks'])

plt.show()
```

The first box plot represents:

```text
Original Data
```

The second box plot represents:

```text
Data After Trimming
```

The purpose of this comparison is to visually check whether the extreme values have been removed.

---

# 15. Technique 2 — Capping

## What is Capping?

**Capping** means replacing outlier values with a predefined upper or lower limit instead of deleting the complete observation.

For example:

```text
10, 12, 11, 13, 12, 100
```

Suppose:

```text
Upper Limit = 90
```

Instead of removing `100`, we replace it with `90`.

The result becomes:

```text
10, 12, 11, 13, 12, 90
```

Therefore:

> **Trimming removes the observation, whereas Capping keeps the observation but changes the extreme value.**

Capping is also commonly known as **Winsorization**.

---

# 16. Why Use Capping?

Capping is useful when the outlier is a genuine observation and we do not want to remove the entire row.

For example, suppose a dataset contains:

```text
50, 52, 51, 49, 53, 100
```

If `100` is a genuine observation, removing the complete row might result in information loss.

Instead, we can cap it at the upper limit.

```text
100 → Upper Limit
```

This keeps the row while reducing the influence of the extreme value.

---

# 17. Capping in This Project

First, we create a copy of the original DataFrame:

```python
new_df_cap = df.copy()
```

This is important because we don't want to directly modify the original DataFrame.

Then we use `np.where()`:

```python
new_df_cap['placement_exam_marks'] = np.where(
    new_df_cap['placement_exam_marks'] > upper_limit,
    upper_limit,

    np.where(
        new_df_cap['placement_exam_marks'] < lower_limit,
        lower_limit,
        new_df_cap['placement_exam_marks']
    )
)
```

Let's understand this step by step.

---

# 18. Understanding np.where()

The basic structure is:

```python
np.where(condition, value_if_true, value_if_false)
```

For example:

```python
np.where(
    marks > upper_limit,
    upper_limit,
    marks
)
```

means:

```text
If marks > upper_limit
        ↓
Replace marks with upper_limit

Otherwise
        ↓
Keep the original marks
```

---

# 19. Understanding Our Capping Code

Our first condition is:

```python
new_df_cap['placement_exam_marks'] > upper_limit
```

If this condition is true:

```python
upper_limit
```

is used.

So:

```text
100 → upper_limit
```

Then we check the lower limit:

```python
new_df_cap['placement_exam_marks'] < lower_limit
```

If this is true:

```python
lower_limit
```

is used.

So:

```text
5 → lower_limit
```

If the value is already within the valid range:

```text
lower_limit <= value <= upper_limit
```

the original value is kept.

---

# 20. Capping Logic

The complete logic can be represented as:

```text
                    Value
                      |
          ┌───────────┴───────────┐
          ↓                       ↓
   Value > Upper Limit     Value < Lower Limit
          ↓                       ↓
   Replace with Upper       Replace with Lower
          Limit                   Limit
          │                       │
          └───────────┬───────────┘
                      ↓
               Otherwise
                      ↓
             Keep Original Value
```

In simple terms:

```text
Too High  → Upper Limit
Too Low   → Lower Limit
Normal    → Original Value
```

---

# 21. Check the Shape After Capping

We can check the number of rows and columns:

```python
new_df_cap.shape
```

One important advantage of capping is that:

> **The number of rows does not decrease.**

Unlike trimming, capping keeps all observations.

For example:

```text
Original Data      → 1000 rows
After Trimming     → 950 rows
After Capping      → 1000 rows
```

The exact numbers depend on the dataset.

---

# 22. Visualizing Capping

We compare the original data with the capped data:

```python
plt.figure(figsize=(15, 8))

plt.subplot(221)

sns.boxplot(x=df['placement_exam_marks'])

plt.subplot(222)

sns.boxplot(x=new_df_cap['placement_exam_marks'])

plt.show()
```

The first box plot shows the original distribution.

The second box plot shows the distribution after capping.

The extreme values are no longer allowed to extend beyond the calculated limits.

---

# 23. Trimming vs Capping

| Feature                         | Trimming            | Capping          |
| ------------------------------- | ------------------- | ---------------- |
| Outlier removed?                | Yes                 | No               |
| Row removed?                    | Yes                 | No               |
| Original extreme value changed? | Row is removed      | Yes              |
| Dataset size                    | Decreases           | Remains the same |
| Information loss                | Possible            | Lower            |
| Useful for genuine outliers     | Sometimes not ideal | Often useful     |
| Common name                     | Trimming            | Winsorization    |

---

# 24. Simple Example

Suppose our data is:

```text
10, 12, 11, 13, 12, 100
```

Assume:

```text
Upper Limit = 90
```

### Trimming

```text
10, 12, 11, 13, 12, 100
                     ↓
                    REMOVE
```

Result:

```text
10, 12, 11, 13, 12
```

### Capping

```text
10, 12, 11, 13, 12, 100
                     ↓
                    90
```

Result:

```text
10, 12, 11, 13, 12, 90
```

So:

```text
TRIMMING
100 → removed

CAPPING
100 → 90
```

---

# 25. Complete Code

```python
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv('newplacementdata.csv')

# Check first few rows
df.head(3)

# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical summary
df['placement_exam_marks'].describe()

# Visualize outliers
sns.boxplot(x=df['placement_exam_marks'])

# Find Q1 and Q3
percentile25 = df['placement_exam_marks'].quantile(0.25)
percentile75 = df['placement_exam_marks'].quantile(0.75)

print("percentile25 is :", percentile25)
print("percentile75 is :", percentile75)

# Calculate IQR
IQR = percentile75 - percentile25

print("IQR:", IQR)

# Calculate upper and lower limits
upper_limit = percentile75 + 1.5 * IQR
lower_limit = percentile25 - 1.5 * IQR

print("Upper Limit:", upper_limit)
print("Lower Limit:", lower_limit)

# Find upper outliers
df[df['placement_exam_marks'] > upper_limit]

# Find lower outliers
df[df['placement_exam_marks'] < lower_limit]


# =====================================================
# 1. TRIMMING
# =====================================================

# Remove upper outliers
newdf = df[df['placement_exam_marks'] < upper_limit]

# Compare before and after trimming
plt.figure(figsize=(15, 5))

plt.subplot(121)
sns.boxplot(x=df['placement_exam_marks'])
plt.title("Before Trimming")

plt.subplot(122)
sns.boxplot(x=newdf['placement_exam_marks'])
plt.title("After Trimming")

plt.show()


# =====================================================
# 2. CAPPING
# =====================================================

# Create a copy of the original DataFrame
new_df_cap = df.copy()

# Cap upper and lower outliers
new_df_cap['placement_exam_marks'] = np.where(
    new_df_cap['placement_exam_marks'] > upper_limit,
    upper_limit,

    np.where(
        new_df_cap['placement_exam_marks'] < lower_limit,
        lower_limit,
        new_df_cap['placement_exam_marks']
    )
)

# Check shape
new_df_cap.shape

# Compare before and after capping
plt.figure(figsize=(15, 8))

plt.subplot(221)
sns.boxplot(x=df['placement_exam_marks'])
plt.title("Before Capping")

plt.subplot(222)
sns.boxplot(x=new_df_cap['placement_exam_marks'])
plt.title("After Capping")

plt.show()
```

---

# 26. Important Correction About Capping

Capping does **not** "expand the range."

Instead, capping **limits or reduces the effect of extreme values** by bringing them back to the upper or lower boundary.

For example:

```text
Original:

5, 10, 15, 20, 30, 50
                    ↑
                  Outlier
```

After capping:

```text
5, 10, 15, 20, 30, Upper Limit
```

So the correct statement is:

> **Capping restricts extreme values to predefined upper and lower limits while keeping the original observations.**

---

# 27. Important Practical Point

We should **not blindly remove every outlier**.

An outlier can be:

```text
Incorrect data
      OR
Genuine extreme observation
```

For example, if the dataset contains salaries:

```text
30,000
32,000
35,000
31,000
30,000
5,00,000
```

`5,00,000` might look like an outlier, but it could be a genuine salary of a senior employee.

Therefore, we should understand the business/data context before deciding whether to:

* Remove the outlier
* Cap the outlier
* Transform the data
* Keep the outlier

---

# 28. Complete Concept in One Flow

```text
                    Dataset
                       ↓
                Identify Outliers
                       ↓
                  IQR Method
                       ↓
              Calculate Q1 and Q3
                       ↓
                  IQR = Q3 - Q1
                       ↓
        ┌──────────────┴──────────────┐
        ↓                             ↓
 Lower = Q1 - 1.5×IQR        Upper = Q3 + 1.5×IQR
        ↓                             ↓
        └──────────────┬──────────────┘
                       ↓
                Handle Outliers
                       ↓
              ┌────────┴────────┐
              ↓                 ↓
          Trimming           Capping
              ↓                 ↓
       Remove the row      Replace extreme
       containing the      value with limit
          outlier
```

---

# 29. Interview Questions

### Q1. What is an outlier?

An outlier is a data point that is significantly different from the majority of observations in a dataset.

### Q2. Why do we handle outliers?

Outliers can affect statistical calculations and the performance of some Machine Learning algorithms. Handling them can help create a more representative dataset.

### Q3. What is IQR?

IQR stands for Interquartile Range.

```text
IQR = Q3 - Q1
```

It represents the middle 50% of the data.

### Q4. How do you detect outliers using IQR?

We calculate:

```text
Lower Limit = Q1 - 1.5 × IQR
Upper Limit = Q3 + 1.5 × IQR
```

Values outside these limits are considered potential outliers.

### Q5. What is trimming?

Trimming is the process of completely removing observations containing outliers.

### Q6. What is capping?

Capping is the process of replacing outlier values with predefined upper or lower limits instead of removing the observations.

### Q7. What is the difference between trimming and capping?

> **Trimming removes the observation, whereas capping keeps the observation but limits the extreme value.**

### Q8. Does capping reduce the number of rows?

No. Capping keeps the observations, so the number of rows generally remains the same.

### Q9. Does trimming reduce the number of rows?

Yes. Since observations containing outliers are removed, the number of rows can decrease.

### Q10. Is every outlier an error?

No. An outlier can be a genuine observation. We should understand the data before deciding how to handle it.

---

# 30. Key Points to Remember

```text
Outlier
   ↓
Unusual/extreme observation

IQR
   ↓
Q3 - Q1

Lower Limit
   ↓
Q1 - 1.5 × IQR

Upper Limit
   ↓
Q3 + 1.5 × IQR

Trimming
   ↓
Remove the observation

Capping
   ↓
Replace extreme value with boundary

Trimming
   ↓
Rows decrease

Capping
   ↓
Rows remain
```

### One-line definition

> **Outlier handling is the process of identifying extreme observations and appropriately treating them using techniques such as trimming or capping to reduce their unwanted influence on data analysis and Machine Learning models.**
