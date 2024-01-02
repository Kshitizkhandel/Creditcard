import pathlib
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml

def load_data(data_path):
    df=pd.read_csv(data_path)
    return df

def split(df,train_size,seed):
    train,test=train_test_split(df,train_size=train_size,random_state=seed)
    return train,test

def save_data(output_path,train,test):
    pathlib.Path(output_path).mkdir(exist_ok=True,parents=True)
    train_data=train.to_csv(output_path + '/train.csv')
    test_data=train.to_csv(output_path + '/test.csv')

def main():
    current_directory=pathlib.Path(__file__)
    print('current',current_directory)
    home_dir=current_directory.parent.parent.parent
    print('home_dir',home_dir)
    params_path=home_dir.as_posix() + '/params.yaml'
    params_file=yaml.safe_load(open(params_path))
    params=params_file['make_dataset']
    print('params',params)
    input_file=home_dir.as_posix() + '/data/raw/creditcard.csv'
    data=load_data(input_file)
    print('data loaded')
    train_data,test_data=split(data,params['test_split'],params['seed'])
    output_path=home_dir.as_posix() + '/data/process_data'
    save_data(output_path,train_data,test_data)

if __name__=='__main__' :
    main()   




