import pathlib
import pandas as pd
import sys
import yaml
from sklearn.model_selection import train_test_split

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def split_data(df,test_size,seed):
    train_split,test_split=train_test_split(df,test_size=test_size,random_state=seed)
    return train_split,test_split

def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)



def main():
    curr_dir=pathlib.Path(__file__)
    print('curr_dir',curr_dir)
    home_dir=curr_dir.parent.parent.parent
    print('home_dir',home_dir)
    print('as posix',home_dir.as_posix())
    params_file=home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["make_dataset"]
    print('params',params)
    input_file='/data/raw/creditcard.csv'
    data_path=home_dir.as_posix() + input_file
    print('data_path',data_path)
    output_path=home_dir.as_posix() + '/data/processed'
    data=load_data(data_path)
    train_data,test_data=split_data(data,params['test_split'],params['seed'])
    save_data(train_data,test_data,output_path)


    
if __name__=='__main__':
    main()



    