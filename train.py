from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingRegressor
from data_model import DataModel
import pickle
import os

# save model
def save(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
# load model
def load(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    
    return model

# train model and make prediction
def train_test(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred, squared=False)


def main():
    # create a folder for models
    if not os.path.exists("models"):
        os.makedirs("models")

    dm = DataModel("data/220511_monatszahlenmonatszahlen2204_verkehrsunfaelle.csv")
    dm.encode_label()

    df_train = dm.df[dm.df["JAHR"] < 2021].reset_index(drop=True)
    df_test = dm.df[dm.df["JAHR"] == 2021].reset_index(drop=True)

    # data preparation
    X_train = df_train.iloc[:,:4].values
    y_train = df_train.iloc[:,-1].values

    X_test = df_test.iloc[:,:4].values
    y_test = df_test.iloc[:,-1].values

    # create models for training
    model_random_forest = RandomForestRegressor(max_depth=12, random_state=2)

    model_voting_random_forest = RandomForestRegressor(max_depth=12, random_state=2)
    model_voting_extra_tree = ExtraTreesRegressor(n_estimators=1, random_state=2)
    model_voting = VotingRegressor(estimators=[('rf', model_voting_random_forest), ('et', model_voting_extra_tree)])

    estimators = [
        ('lr', GradientBoostingRegressor(random_state=2)),
        ('svr', ExtraTreesRegressor(n_estimators=32, random_state=2))
        ]
    model_stack = StackingRegressor(
        estimators=estimators,
        final_estimator=RandomForestRegressor(max_depth=12, random_state=2))

    # calculate the rmse score and save the model
    mse_rf = train_test(model_random_forest, X_train, y_train, X_test, y_test)
    save(model_random_forest, "models/random_forest.pickle")

    mse_voting = train_test(model_voting, X_train, y_train, X_test, y_test)
    save(model_voting, "models/voting_rf_et.pickle")

    mse_stack = train_test(model_stack, X_train, y_train, X_test, y_test)
    save(model_stack, "models/stacking_reg.pickle")


    print("-"*15, "Mean Squared Error Table", "-"*15)
    print("Random Forest: {:.4f}".format(mse_rf))
    print("Voting       : {:.4f}".format(mse_voting))
    print("Stacking     : {:.4f}".format(mse_stack))


if __name__ == "__main__":
    main()

