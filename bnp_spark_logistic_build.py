import pandas as pd
from pyspark.ml.classification import *
import bnp_helper
import common_helper
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn import metrics   #Additional scklearn functions


train_df_raw = pd.read_csv("../data/train.csv")
test_df_raw = pd.read_csv("../data/test.csv")


df_raw_combined = pd.concat([train_df_raw, test_df_raw], axis = 0)

########### Clean and Impute (Combined) #############
df_combinded = bnp_helper.clean(df_raw_combined, drop_collinearity = True, inplace = True)
df_combinded = bnp_helper.impute_cate_with_na_numeric_with_outlier(df_combinded)

################ Convert text to number (Combined) ################
df_combinded = common_helper.dummify(df_combinded, bnp_helper.get_categorical_variables(df_combinded))
print df_combinded.shape
print df_combinded.head()

train_df = df_combinded[-df_raw_combined['target'].isnull()]
test_df = df_combinded[df_raw_combined['target'].isnull()]

train_df_sample = train_df.sample(5000, random_state = 0)
target_train = train_df_sample['target']
train_data = train_df_sample.drop(['ID'], axis = 1)

train_data = sqlContext.createDataFrame(train_data, list(train_data.columns))

assembler = VectorAssembler(inputCols=list(train_data.columns), outputCol='features')

train_data = assembler.transform(train_data)

lr = LogisticRegression(labelCol="target")

model = lr.fit(train_data)

prediction = model.transform(train_data)

evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC", labelCol="target")
print "ROC score: {}".format(evaluator.evaluate(prediction))

log_loss = metrics.log_loss(target_train, list(prediction.probability))
print "log loss: {}".format(log_loss)
 




