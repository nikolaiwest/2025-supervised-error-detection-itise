| Classifier Name                | Library | Family        | Debug | Fast | Paper | sklearn | sktime | Full |
|--------------------------------|---------|---------------|-------|------|-------|---------|--------|------|
| DummyClassifier                | sklearn | Benchmark     |   X   |      |   X   |    X    |        |   X  |
| LogisticRegression             | sklearn | Linear        |       |   X  |       |    X    |        |   X  |
| LinearSVC                      | sklearn | Linear        |       |      |       |    X    |        |   X  |
| SGDClassifier                  | sklearn | Linear        |       |      |       |    X    |        |   X  |
| RandomForestClassifier         | sklearn | Tree          |       |      |   X   |    X    |        |   X  |
| GradientBoostingClassifier     | sklearn | Tree          |       |      |       |    X    |        |   X  |
| BaggingClassifier              | sklearn | Tree          |       |      |       |    X    |        |   X  |
| SupportVectorClassifier (SVC)  | sklearn | Kernel        |       |      |   X   |    X    |        |   X  |
| KNeighborsClassifier           | sklearn | Distance      |       |      |       |    X    |        |   X  |
| NearestCentroid                | sklearn | Distance      |       |      |       |    X    |        |   X  |
| GaussianNB                     | sklearn | Probabilistic |       |      |       |    X    |        |   X  |
| QuadraticDiscriminantAnalysis  | sklearn | Probabilistic |       |      |       |    X    |        |   X  |
| MLPClassifier                  | sklearn | Neural        |       |      |       |    X    |        |   X  |
| KNeighborsTimeSeriesClassifier | sktime  | Distance      |       |   X  |       |         |    X   |   X  |
| ElasticEnsemble                | sktime  | Distance      |       |      |       |         |    X   |   X  |
| ProximityForest                | sktime  | Distance      |       |      |       |         |    X   |   X  |
| TimeSeriesForestClassifier     | sktime  | Interval      |       |      |   X   |         |    X   |   X  |
| RandomIntervalSpectralEnsemble | sktime  | Interval      |       |      |       |         |    X   |   X  |
| CanonicalIntervalForest        | sktime  | Interval      |       |      |       |         |    X   |   X  |
| BOSSEnsemble                   | sktime  | Dictionary    |       |      |       |         |    X   |   X  |
| WEASEL                         | sktime  | Dictionary    |       |      |       |         |    X   |   X  |
| ShapeletTransformClassifier    | sktime  | Shapelet      |       |      |   X   |         |    X   |   X  |
| ShapeDTW                       | sktime  | Shapelet      |       |      |       |         |    X   |   X  |
| ROCKET                         | sktime  | Advanced      |       |      |       |         |    X   |   X  |
| Arsenal                        | sktime  | Advanced      |       |      |       |         |    X   |   X  |