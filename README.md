# spark-sklearn-airbnb-predict
Code example to predict prices of Airbnb vacation rentals, using scikit-learn on Spark.

[The Jupyter notebook](https://github.com/OElesin/spark-sklearn-airbnb-predict/blob/master/Predicting%20Airbnb%20Listing%20with%20Spark%20and%20Scikit-Learn.ipynb) in this repo contains examples to run regression estimators on the [Inside Airbnb](http://insideairbnb.com/get-the-data.html) listings dataset from Amsterdam. The target variable is the price of the listing. To speed up the hyperparameter search, the notebook shows examples that use the spark-sklearn package to distribute GridSearchCV across nodes in a Spark cluster. This provides a much faster way to search and can lead to better results.

To run the scikit-learn examples (without Spark) the following packages are required:

- Python 2
- Pandas
- NumPy
- scikit-learn (0.17 or later)
These can be installed on the machine.

To run the [a spark-sklearn](https://github.com/databricks/spark-sklearn) examples with Spark, the following packages are required on each machine:

All of the above packages
Spark (1.5 or later) 

Note that the current spark-sklearn requires Spark 2.0. However your can edit **spark_sklearn/converter.py** and **spark_sklearn/util.py** for earlier versions of Spark

[spark-sklearn](https://github.com/databricks/spark-sklearn) -- follow the installation instructions there

*Happy Doing Data*

***Olalekan Elesin***
