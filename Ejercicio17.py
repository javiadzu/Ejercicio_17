{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd # para leer datos\n",
    "import sklearn.ensemble # para el random forest\n",
    "import requests\n",
    "import sklearn.model_selection # para split train-test\n",
    "import sklearn.metrics # para calcular el f1-score\n",
    "from sklearn.feature_selection import VarianceThreshold\n",
    "\n",
    "from scipy.io import arff\n",
    "import pandas as pd\n",
    "data = arff.loadarff('1year.arff')\n",
    "df = pd.DataFrame(data[0])\n",
    "predictors = list(data.keys())\n",
    "\n",
    "#Retiro las columnas que tienen '?' p 'NaN'\n",
    "l=df.isin(['?']).any()\n",
    "x = df.isin(['NaN']).any()\n",
    "df.drop([l], axis=1)\n",
    "df.drop([l], axis=1)\n",
    "\n",
    "#Realizamos la partición de los validación, test y training\n",
    "X_train, X_valtest, y_train, y_valtest = sklearn.model_selection.train_test_split(\n",
    "                                    data[predictors], data['class'], test_size=0.5)\n",
    "X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(\n",
    "                                     X_valtest, y_valtest, test_size=0.4)\n",
    "clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, max_features='sqrt')\n",
    "\n",
    "n_trees = np.arange(1,400,25)\n",
    "f1_train = []\n",
    "f1_val = []\n",
    "feature_importance = np.zeros((len(n_trees), len(predictors)))\n",
    "\n",
    "#Realizamos la elección del mejor número de árboles y buscamos los mejores Features\n",
    "for i, n_tree in enumerate(n_trees):\n",
    "    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_tree, max_features='sqrt')\n",
    "    #Para el número de árboles usamos xtrain y xtest\n",
    "    clf.fit(X_train, y_train)\n",
    "     #Para el número de features usamos xtrain y xval\n",
    "    f1_train.append(sklearn.metrics.f1_score(y_val, clf.predict(X_train)))\n",
    "    f1_val.append(sklearn.metrics.f1_score(y_val, clf.predict(X_val)))\n",
    "    feature_importance[i, :] = clf.feature_importances_"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
