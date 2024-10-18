def info(data):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_cols = data.select_dtypes(include=numerics).columns.tolist()

    summary = data[numerical_cols].describe()
    print("Summary Statistics:\n", summary)

    variability = data[numerical_cols].var()
    std_dev = data[numerical_cols].std()

    print("Variance:\n", variability)
    print("Standard Deviation:\n", std_dev)

def check_missing(data):
    missing_values = data.isnull().sum()

    print("\nMissing Values:\n", missing_values[missing_values>0])