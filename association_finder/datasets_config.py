housing_config = {
    "train_dataset_path": "datasets/houseprices/train.csv",
    "index_col": "Id",
    "target_column": "SalePrice",
    "good_columns": ["OverallQual", "YearBuilt", "YearRemodAdd", "OverallCond", "BldgType", "LotArea",
                     "GrLivArea", "FullBath", "BedroomAbvGr", "LotFrontage", "TotalBsmtSF", "SalePrice"],
    "one_hot_columns": ["BldgType"],
    "min_confidence": 0.4,
    "min_support": 0.4,
    "diff_threshold": 0.1
}

rain_config = {
    "train_dataset_path": "datasets/rain_in_australia/weatherAUS.csv",
    "index_col": "Date",
    "target_column": "RainTomorrow",
    "good_columns": ['Location', 'MinTemp', 'MaxTemp',  # 'Rainfall',
                     # 'Evaporation',
                     'Sunshine', 'WindGustDir', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                     'WindGustSpeed',
                     # 'Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm', 'WindDir9am', 'WindDir3pm',
                     'RainToday', 'RainTomorrow'],
    "one_hot_columns": ["Location", "WindGustDir"],
    "min_confidence": 0.4,
    "min_support": 0.4,
    "diff_threshold": 0.1
}

sales_config = {
    "train_dataset_path": "datasets/big_mart_sales/Train-Set.csv",
    "index_col": "ProductID",
    "target_column": "OutletSales",
    "good_columns": ['Weight', 'FatContent', 'ProductVisibility', 'ProductType', 'MRP',
                     'OutletID', 'EstablishmentYear', 'OutletSize', 'LocationType', 'OutletType', 'OutletSales'],
    "one_hot_columns": ['FatContent','ProductType','OutletID','OutletSize','LocationType','OutletType'],
    "min_confidence": 0.1,
    "min_support": 0.1,
    "diff_threshold": 0.1
}

netflix_config = {
    "train_dataset_path": "datasets/netflix_data/netflix-rotten-tomatoes-metacritic-imdb.csv",
    "index_col": "Title",
    "target_column": "Hidden Gem Score",
    "good_columns": ['Genre', 'Languages', 'Series or Movie', 'Hidden Gem Score','Country Availability','Runtime',
                'Director', 'Writer', 'View Rating', 'IMDb Score', 'Rotten Tomatoes Score','Metacritic Score',
               'Awards Received','Awards Nominated For','Release Date','Netflix Release Date','IMDb Votes'],
    "one_hot_columns": ['Genre','Languages','Series or Movie','Country Availability','Runtime','Director','Writer','View Rating',
                  'Release Date','Netflix Release Date'],
    "min_confidence": 0.4,
    "min_support": 0.4,
    "diff_threshold": 0.1
}

datasets_config = {
    "housing": housing_config,
    "rain": rain_config,
    "sales": sales_config,
    "netflix": netflix_config
}
