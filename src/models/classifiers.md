| Classifier Name                | Library | Family        | Debug | Fast | Paper | sklearn | sktime | Full |
|--------------------------------|---------|---------------|-------|------|-------|---------|--------|------|
| DummyClassifier                | sklearn | Benchmark     |   X   |   X  |   X   |    X    |        |   X  |
| LogisticRegression             | sklearn | Linear        |       |   X  |       |    X    |        |   X  |
| LinearSVC                      | sklearn | Linear        |       |      |       |    X    |        |   X  |
| SGDClassifier                  | sklearn | Linear        |       |      |       |    X    |        |   X  |
| DecisionTreeClassifier         | sklearn | Tree          |       |   X  |       |         |        |      |
| RandomForestClassifier         | sklearn | Tree          |       |      |   X   |    X    |        |   X  |
| ExtraTreesClassifier           | sklearn | Tree          |       |      |       |         |        |      |
| GradientBoostingClassifier     | sklearn | Tree          |       |      |       |    X    |        |   X  |
| BaggingClassifier              | sklearn | Tree          |       |      |       |    X    |        |   X  |
| SVC                            | sklearn | Kernel        |       |      |   X   |    X    |        |   X  |
| KNeighborsClassifier           | sklearn | Distance      |       |      |       |    X    |        |   X  |
| NearestCentroid                | sklearn | Distance      |       |      |       |         |        |      |
| GaussianNB                     | sklearn | Probabilistic |       |      |       |    X    |        |   X  |
| QuadraticDiscriminantAnalysis  | sklearn | Probabilistic |       |      |       |    X    |        |   X  |
| MLPClassifier                  | sklearn | Neural        |       |      |       |    X    |        |   X  |
| KNeighborsTimeSeriesClassifier | sktime  | Distance      |       |      |       |         |    X   |   X  |
| ElasticEnsemble                | sktime  | Distance      |       |      |       |         |        |      |
| ProximityForest                | sktime  | Distance      |       |      |       |         |        |      |
| TimeSeriesForestClassifier     | sktime  | Interval      |       |      |   X   |         |    X   |   X  |
| RandomIntervalSpectralEnsemble | sktime  | Interval      |       |      |       |         |    X   |   X  |
| CanonicalIntervalForest        | sktime  | Interval      |       |      |       |         |    X   |   X  |
| DrCIF                          | sktime  | Interval      |       |      |       |         |        |      |
| BOSSEnsemble                   | sktime  | Dictionary    |       |      |       |         |    X   |   X  |
| WEASEL                         | sktime  | Dictionary    |       |      |       |         |    X   |   X  |
| ShapeletTransformClassifier    | sktime  | Shapelet      |       |      |       |         |    X   |   X  |
| ShapeDTW                       | sktime  | Shapelet      |       |      |       |         |    X   |   X  |
| ROCKET                         | sktime  | Advanced      |       |      |   X   |         |    X   |   X  |
| Catch22Classifier              | sktime  | Feature       |   X   |      |       |         |    X   |   X  |
| Arsenal                        | sktime  | Advanced      |       |      |       |         |    X   |   X  |
| MiniRocket                     | sktime  | Advanced      |       |      |       |         |    X   |   X  |