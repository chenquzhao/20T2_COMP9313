from pyspark.sql.functions import rand
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer


def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category", output_feature_col="features", output_label_col="label"):
    word_tokenizer = Tokenizer(inputCol=input_descript_col, outputCol="words")
    count_vectorizer = CountVectorizer(inputCol="words", outputCol=output_feature_col)
    label_indexer = StringIndexer(inputCol=input_category_col, outputCol=output_label_col)

    # pick output columns
    class Formatter(Transformer):
        def __init__(self, outputCols):
            self.outputCols = outputCols

        def _transform(self, df):
            return df.select(*self.outputCols)

    formatter = Formatter(outputCols=['id', 'features', 'label'])
    base_features_pipeline = Pipeline(stages=[word_tokenizer, count_vectorizer, label_indexer, formatter])
    return base_features_pipeline


def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):
    meta_features = None

    for i in range(5):
        # decide train set and test set
        condition = training_df['group'] == i
        c_train = training_df.filter(~condition).cache()
        c_test = training_df.filter(condition).cache()

        nb_model_0 = nb_0.fit(c_train)
        nb_pred_0 = nb_model_0.transform(c_test)
        nb_pred_0 = nb_pred_0.drop('nb_raw_0', 'nb_prob_0')

        nb_model_1 = nb_1.fit(c_train)
        nb_pred_1 = nb_model_1.transform(c_test).select('id', 'nb_pred_1')

        nb_model_2 = nb_2.fit(c_train)
        nb_pred_2 = nb_model_2.transform(c_test).select('id', 'nb_pred_2')

        svm_model_0 = svm_0.fit(c_train)
        svm_pred_0 = svm_model_0.transform(c_test).select('id', 'svm_pred_0')

        svm_model_1 = svm_1.fit(c_train)
        svm_pred_1 = svm_model_1.transform(c_test).select('id', 'svm_pred_1')

        svm_model_2 = svm_2.fit(c_train)
        svm_pred_2 = svm_model_2.transform(c_test).select('id', 'svm_pred_2')

        df = nb_pred_0.join(nb_pred_1, 'id', "right_outer").join(nb_pred_2, 'id', "right_outer")
        df = df.join(svm_pred_0, 'id', "right_outer").join(svm_pred_1, 'id', "right_outer").join(svm_pred_2, 'id',
                                                                                                 "right_outer")

        # compute joint prediction
        df = df.withColumn('joint_pred_0', df.nb_pred_0 * 2 + df.svm_pred_0)
        df = df.withColumn('joint_pred_1', df.nb_pred_1 * 2 + df.svm_pred_1)
        df = df.withColumn('joint_pred_2', df.nb_pred_2 * 2 + df.svm_pred_2)
        if i == 0:
            meta_features = df
        else:
            meta_features = meta_features.union(df)

    return meta_features


def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier):
    rseed = 1024

    def gen_binary_labels(df):
        df = df.withColumn('label_0', (df['label'] == 0).cast(DoubleType()))
        df = df.withColumn('label_1', (df['label'] == 1).cast(DoubleType()))
        df = df.withColumn('label_2', (df['label'] == 2).cast(DoubleType()))
        return df

    # prepare test set
    test_set = base_features_pipeline_model.transform(test_df)
    test_set = test_set.withColumn('group', (rand(rseed) * 5).cast(IntegerType()))

    # generate base features
    test_set = gen_binary_labels(test_set)
    test_set = gen_base_pred_pipeline_model.transform(test_set)

    # generate meta features
    test_set = test_set.withColumn('joint_pred_0', test_set.nb_pred_0 * 2 + test_set.svm_pred_0)
    test_set = test_set.withColumn('joint_pred_1', test_set.nb_pred_1 * 2 + test_set.svm_pred_1)
    test_set = test_set.withColumn('joint_pred_2', test_set.nb_pred_2 * 2 + test_set.svm_pred_2)
    test_set = gen_meta_feature_pipeline_model.transform(test_set)

    # predict using meta classifier
    test_set = meta_classifier.transform(test_set)
    test_set = test_set.select('id', 'label', 'final_prediction')

    return test_set
