from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,r2_score
import mlflow
import mlflow.sklearn

mlflow.set_experiment("Diabetes_Model_Comparison")



load=load_diabetes()

X=load.data
y=load.target

X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.2,
                                               random_state=420
                                               
                                               )


model=LinearRegression()
model2=Ridge()

# training
model.fit(X_train,y_train)
model2.fit(X_train,y_train)

# prediction
prediction=model.predict(X_test)
prediction2=model2.predict(X_test)

# metrics

mae=mean_absolute_error(y_test,prediction)
mae_ridge=mean_absolute_error(y_test,prediction2)



print("mae_for_lr",mae)
print("mae_for_ridge",mae_ridge)

r2=r2_score(y_test,prediction)
r2_ridge=r2_score(y_test,prediction2)

print("r2 for lr",r2)
print("r2 for ridge",r2_ridge)


with mlflow.start_run():
    mlflow.log_param("model","LinearRegression")
    mlflow.log_metric("MAE",mae)
    mlflow.log_metric("R2",r2)
    mlflow.sklearn.log_model(sk_model=model,name="linear_regression_model")    

with mlflow.start_run  ():
    mlflow.log_param("model_ridge","Ridge")
    mlflow.log_metric("MAE",mae_ridge)
    mlflow.log_metric("R2",r2_ridge) 
    mlflow.sklearn.log_model(sk_model=model2,name="ridge_model")     
model_uri = "runs:/3f822d3e4dff48b988f95798a87a4c9d/ridge_model"

mlflow.register_model(
    model_uri=model_uri,
    name="Diabetes-Prediction-Model"
)
model_uri = "models:/Diabetes-Prediction-Model/1"
loaded_model=mlflow.sklearn.load_model(model_uri)


loaded_model_prediction=loaded_model.predict(X_test)
print(loaded_model_prediction[:5])




print("mae comparision",mean_absolute_error(y_test,loaded_model_prediction))