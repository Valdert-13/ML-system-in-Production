import time
from dataset_builder import time_format
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def prepare_dataset(dataset, mode, dataset_path, INTER_LIST):
    start_t = time.time()
    print('Dealing with missing values, outliers, categorical features...')

    # Профили
    dataset['age'] = dataset['age'].fillna(dataset['age'].median())
    dataset['gender'] = dataset['gender'].fillna(dataset['gender'].mode()[0])
    dataset.loc[~dataset['gender'].isin(['M', 'F']), 'gender'] = dataset['gender'].mode()[0]
    dataset['gender'] = dataset['gender'].map({'M': 1., 'F':0.})
    dataset.loc[(dataset['age'] > 80) | (dataset['age'] < 7), 'age'] = round(dataset['age'].median())
    dataset.loc[dataset['days_between_fl_df'] < -1, 'days_between_fl_df'] = -1
    # Пинги
    for period in range(1,len(INTER_LIST)+1):
        col = 'avg_min_ping_{}'.format(period)
        dataset.loc[(dataset[col] < 0) |
                    (dataset[col].isnull()), col] = dataset.loc[dataset[col] >= 0][col].median()
    # Сессии и прочее
    dataset.fillna(0, inplace=True)
    dataset.to_csv('{}dataset_{}.csv'.format(dataset_path, mode), sep=';', index=False)

    print('Dataset is successfully prepared and saved to {}, run time (dealing with bad values): {}'. \
          format(dataset_path, time_format(time.time()-start_t)))
    return dataset

def scale_balance (dataset):
    X_train = dataset.drop(['user_id', 'is_churned'], axis=1)
    y_train = dataset['is_churned']

    scaler =  MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_train_balanced, y_train_balanced = SMOTE(sampling_strategy=0.3, random_state=42).\
        fit_sample(X_train_scaled, y_train.values)

    X_train, X_test, y_train, y_test = train_test_split(X_train_balanced,
                                                        y_train_balanced,
                                                        test_size=0.3,
                                                        shuffle=True,
                                                        stratify = y_train_balanced,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def scale_test (dataset):
    test = dataset.drop(['user_id'], axis=1)
    scaler = MinMaxScaler()
    test = scaler.fit_transform(test)
    return test