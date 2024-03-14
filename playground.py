# models = 'LinearRegression, GeneralizedLinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor, IsotonicRegression, AFTSurvivalRegression, FMRegressor'
#
# for model in models.split(', '):
#     with open('regression_template.py', 'r') as f:
#         content = f.read()
#         content = content.replace('{{model}}', model)
#
#     with open(f'scripts/{model}.py', 'w') as f:
#         f.write(content)


# models = 'LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, GBTClassifier, MultilayerPerceptronClassifier, LinearSVC, NaiveBayes, FMClassifier'
#
# for model in models.split(', '):
#     with open('classification_template.py', 'r') as f:
#         content = f.read()
#         content = content.replace('{{model}}', model)
#
#     with open(f'scripts/{model}.py', 'w') as f:
#         f.write(content)


models = 'KMeans, GaussianMixture, PowerIterationClustering, LDA, BisectingKMeans'

for model in models.split(', '):
    with open('clustering_template.py', 'r') as f:
        content = f.read()
        content = content.replace('{{model}}', model)

    with open(f'scripts/{model}.py', 'w') as f:
        f.write(content)
