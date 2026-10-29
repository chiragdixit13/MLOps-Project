from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,r2_score
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score

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
# 5-fold cross-validation

lr_score=cross_val_score(
    model,X_train,y_train,cv=5,scoring="neg_mean_absolute_error"
)
ridge_score=cross_val_score(
    model2,X_train,y_train,cv=5,scoring="neg_mean_absolute_error"
)
lr_cv_mae=-lr_score.mean()
ridge_cv_mae=-ridge_score.mean()

print("Linear Regression CV MAE:",lr_cv_mae)
print("Ridge cv mae",ridge_cv_mae)

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


# 1. Select using cross-validation
if lr_cv_mae < ridge_cv_mae:
    best_model = model
    best_model_name = "LinearRegression"
    best_cv_mae = lr_cv_mae
    best_mae = mae
    best_r2 = r2
else:
    best_model = model2
    best_model_name = "Ridge"
    best_cv_mae = ridge_cv_mae
    best_mae = mae_ridge
    best_r2 = r2_ridge

# 2. Performance gate using held-out test metrics
if best_mae < 45 and best_r2 > 0.40:
    print("Model passed performance checks")
else:
    raise ValueError("Model failed performance checks")

# 3. Log model selection and evaluation in MLflow
with mlflow.start_run() as run:
    mlflow.log_param("model", best_model_name)
    mlflow.log_metric("CV_MAE", best_cv_mae)
    mlflow.log_metric("Test_MAE", best_mae)
    mlflow.log_metric("Test_R2", best_r2)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="best_model"
    )

run_id=run.info.run_id

model_uri=f"runs:/{run_id}/best_model"
registered_model=mlflow.register_model(
    model_uri=model_uri,
    name="Diabetes-Prediction-Model"
)
model_uri = f"models:/Diabetes-Prediction-Model/{registered_model.version}"
loaded_model=mlflow.sklearn.load_model(model_uri)


loaded_model_prediction=loaded_model.predict(X_test)
print(loaded_model_prediction[:5])




print("mae comparision",mean_absolute_error(y_test,loaded_model_prediction))