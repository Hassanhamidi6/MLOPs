import os 
import sys
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature   
import logging 


logging.basicConfig(level=logging.WARN)
logger= logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse= np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae , r2






if __name__ == "__main__":
    
    # Data  Ingestion readifn the dataset - Wine quality dataset 

    csv_url =("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv")

    try:
        data= pd.red_csv(csv_url, sep=";")
    except Exception as e :
        logger.exception("Unable to download the data ")


    # Train and Test split 

    train, test = train_test_split(data)

    train_x= train.drop(['quality'],axis=1)
    test_x = test.drop(['quality'],axis=1)
    train_y= train[['quality']]
    test_y = test[['quality']]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        lr.fit(train_x, train_y)

        predict_qualties = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predict_qualties)

        print("Elasticent model (alpha= {:f}):".format(alpha, l1_ratio))
        print(" RMSE: %s" % rmse)
        print(" MAE: %s" % mae)
        print(" R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("rmse", rmse)
        mlflow.log_param("mae", mae)
        mlflow.log_param("r2", r2)
        
        # for the remote server AWS we neeed to do the setup

        remote_server_uri= ""
        mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.load_model(
                lr, "model", registered_model_name="ElasticnetWithModel"
            )
        else:
            mlflow.sklearn.load_model(lr, "model")