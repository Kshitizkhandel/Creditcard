from sklearn.ensemble import RandomForestClassifier
import joblib
import pathlib
import yaml
import pandas as pd

def train_model(train_features,label,max_depth,n_estimators,seed):
    model=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=seed)
    model.fit(train_features,label)
    return model

def save_model(output_path,model):
    joblib.dump(model,output_path+'/model.joblib')

def main():
    current_directory=pathlib.Path(__file__)
    home_directory=current_directory.parent.parent.parent
    params_path=home_directory.as_posix() + '/params.yaml'
    params_file=yaml.safe_load(open(params_path))
    print("params_file",params_file)
    params=params_file['train_model']
    input_file=home_directory.as_posix() +'/data/processed/train.csv'
    df=pd.read_csv(input_file)
    X=df.drop('Class',axis=1)
    Y=df['Class']
    print("Shape of X",X.shape)
    print("Shape of Y",Y.shape)
    model=train_model(X,Y,params['max_depth'],params['n_estimators'],params['seed'])
    output_path=home_directory.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(exist_ok=True,parents=True)
    save_model(output_path,model)
    print("Model loaded")

if __name__=="__main__":
    main()    



    