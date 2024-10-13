# Problematic Internet Usage Prediction

### Overview
This project focuses on predicting the level of problematic internet usage exhibited by children and adolescents based on their physical activity and fitness data. Given the increasing prevalence of internet usage among youth, understanding and predicting problematic patterns is crucial for intervention and support strategies.

### Problem Statement
The rise of the internet has brought significant benefits, but it has also led to concerning trends in problematic internet usage, particularly among children and adolescents. This project seeks to identify and predict these problematic usage patterns through machine learning techniques, allowing for proactive measures to be implemented by parents, educators, and health professionals.

### Objective
The primary objective of this project is to build a predictive model that accurately assesses the severity of problematic internet usage based on various physical activity and fitness metrics. This involves:
- Data collection and preprocessing
- Feature engineering
- Training multiple machine learning models
- Evaluating model performance using appropriate metrics
- Hyperparameter tuning to optimize model accuracy

### Table of Contents
1. [Technologies](#technologies)
2. [Data Sources](#data-sources)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Training](#model-training)
7. [Results](#results)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Stacked Ensemble Techniques](#stacked-ensemble-techniques)
10. [Conclusion](#conclusion)
11. [Contributing](#contributing)
12. [License](#license)

### Technologies
This project is developed using the following technologies:
- **Python**: Programming language used for data manipulation and machine learning.
- **Pandas**: Library for data manipulation and analysis.
- **NumPy**: Library for numerical computations.
- **Scikit-learn**: Machine learning library for model training and evaluation.
- **XGBoost**: Optimized gradient boosting library.
- **LightGBM**: Fast gradient boosting framework that uses tree-based learning algorithms.
- **CatBoost**: Gradient boosting library that handles categorical features well.
- **Optuna**: Hyperparameter optimization framework.
- **Seaborn**: Data visualization library based on Matplotlib.
- **Matplotlib**: Library for creating static, animated, and interactive visualizations in Python.
- **tqdm**: Library for displaying progress bars.

### Data Sources
The dataset is sourced from the Child Mind Institute Problematic Internet Use Dataset. The following files are utilized:
- `train.csv`: Contains training data for model building.
- `test.csv`: Contains test data for model evaluation.
- `sample_submission.csv`: Format for submitting predictions.
- Time series data in parquet format.

### Installation
To run this project, you need to install the required packages. You can do this using pip:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost seaborn matplotlib optuna tqdm
```

### Usage
1. **Load the Data**: The dataset is loaded using pandas.
2. **Data Preprocessing**: Missing values are handled, and relevant features are created and encoded.
3. **Model Training**: Different models are trained, and their performance is evaluated using the Quadratic Weighted Kappa metric.
4. **Hyperparameter Tuning**: Optuna is used for hyperparameter optimization.


### Data Preprocessing
Data preprocessing is crucial for ensuring the quality and effectiveness of machine learning models. The following steps were taken to prepare the data:
1. **Data Cleaning**
   - Missing values were identified and handled appropriately. For instance, specific physical attributes such as Physical-BMI, Physical-Height, Physical-Weight, Physical-Diastolic_BP, Physical-HeartRate, and Physical-Systolic_BP were imputed using the Basic_Demos-Age and Basic_Demos-Sex columns.

2. **Handling Missing Values**
   - Identifying Missing Values: Assess missing values across different features.
   - Dropping Columns: Features with more than 50% missing values are dropped.
   -	Missing values in the specified physical attributes were imputed based on age and sex groups, creating categories such as ‘Children (5-12)’, ‘Adolescents (13-18)’, and ‘Adults (19-22)’.
	 -	The PreInt_EduHx-computerinternet_hoursday feature was imputed using the Severity Impairment Index (SII).

3. **Feature Engineering**
   - Creating New Features: Aggregate scores derived from existing features enhance dataset informativeness.
     
4. **Encoding Categorical Variables**
   - Categorical features were transformed into numeric values using one-hot encoding to ensure compatibility with machine learning algorithms.

5. **Data Normalization**
   - Feature scaling was performed to standardize the range of independent variables, which is especially important for distance-based algorithms.
     
6. **Splitting Data**
   - The dataset is divided into features (X) and target variable (y), further splitting into training and validation sets.

#### Key Functions
- `load_time_series(dirname)`: Loads time series data from specified directories.
- `dropMissingValueFeatures(train, test)`: Drops features with excessive missing values from datasets.
- `imputing(df)`: Fills in missing values using appropriate methods.

### Model Training
The following models are implemented:
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost
- LightGBM
- CatBoost
- Stacked Ensemble Techniques

### Evaluation
Model performance is evaluated using the Quadratic Weighted Kappa metric, measuring agreement between predicted and actual values while accounting for chance agreements.

### Results
Upon training the models, the following average scores are obtained:
| Model               | Average Quadratic Weighted Kappa |
|---------------------|----------------------------------|
| Decision Tree       | 0.5020                           |
| Random Forest       | 0.6656                           |
| XGBoost             | 0.6402                           |
| LightGBM            | 0.6486                           |
| CatBoost            | 0.6530                           |
| Stacked Ensemble Techniques (Best)            | 0.6624                           |

***Random Forest is choosen as the final model.***


### Hyperparameter Tuning
Optuna is used for hyperparameter optimization involving:
1. Defining an objective function for model training and evaluation.
2. Specifying search space for hyperparameters.
3. Running optimization to find best parameter values.

### Stacked Ensemble Techniques
Stacked ensemble techniques involve:
1. Training diverse base models (e.g., Decision Trees, Random Forests).
2. Training a meta-model on predictions of base models to enhance accuracy.
3. Using cross-validation to prevent overfitting during meta-model training.
4. Evaluating stacked model performance against individual base models.

### Conclusion
This project successfully developed a machine learning model to predict problematic internet usage levels based on physical activity and fitness data. The results demonstrate the effectiveness of the selected algorithms. Random forest after hyperparameter tuning performed the best and is selected for the final prediction.

### Contributing
If you’d like to contribute to this project, please fork the repository and submit a pull request with your changes. Contributions are welcome :) 

### License
This project is licensed under the Apache-2.0 license. See the [LICENSE](LICENSE) file for more details.


***Feel free to modify any sections or add additional details as needed to ensure it accurately reflects your project!***
