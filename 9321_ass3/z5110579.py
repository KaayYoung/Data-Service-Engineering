import sys
import sklearn
import pandas as pd
import xgboost as xgb
import json
import numpy as np
from collections import Counter


# Extract the month and year
def date_features(df):
    df[['release_month','release_day','release_year']] = df['release_date'].str.split('-',expand=True).replace(np.nan, 0).astype(int)
    df['release_year'] = df['release_year']
    df.loc[ (df['release_year'] <= 18) & (df['release_year'] < 100), "release_year"] += 2000
    df.loc[ (df['release_year'] > 18)  & (df['release_year'] < 100), "release_year"] += 1900
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_month'] = df['release_date'].dt.month
    df.drop(columns=['release_day'], inplace=True)
    df.drop(columns=['release_date'], inplace=True)

    return df


if __name__ == '__main__':
    
    df_training = pd.read_csv('training.csv')
    df_validation = pd.read_csv('validation.csv')

    # runtime
    # runtime_graph(df_training)
    df_training['runtime_cat_min_60'] = df_training['runtime'].apply(lambda x: 1 if (x <=60) else 0)
    df_training['runtime_cat_61_80'] = df_training['runtime'].apply(lambda x: 1 if (x >60)&(x<=80) else 0)
    df_training['runtime_cat_81_100'] = df_training['runtime'].apply(lambda x: 1 if (x >80)&(x<=100) else 0)
    df_training['runtime_cat_101_120'] = df_training['runtime'].apply(lambda x: 1 if (x >100)&(x<=120) else 0)
    df_training['runtime_cat_121_140'] = df_training['runtime'].apply(lambda x: 1 if (x >120)&(x<=140) else 0)
    df_training['runtime_cat_141_170'] = df_training['runtime'].apply(lambda x: 1 if (x >140)&(x<=170) else 0)
    df_training['runtime_cat_171_max'] = df_training['runtime'].apply(lambda x: 1 if (x >=170) else 0)

    df_validation['runtime_cat_min_60'] = df_validation['runtime'].apply(lambda x: 1 if (x <=60) else 0)
    df_validation['runtime_cat_61_80'] = df_validation['runtime'].apply(lambda x: 1 if (x >60)&(x<=80) else 0)
    df_validation['runtime_cat_81_100'] = df_validation['runtime'].apply(lambda x: 1 if (x >80)&(x<=100) else 0)
    df_validation['runtime_cat_101_120'] = df_validation['runtime'].apply(lambda x: 1 if (x >100)&(x<=120) else 0)
    df_validation['runtime_cat_121_140'] = df_validation['runtime'].apply(lambda x: 1 if (x >120)&(x<=140) else 0)
    df_validation['runtime_cat_141_170'] = df_validation['runtime'].apply(lambda x: 1 if (x >140)&(x<=170) else 0)
    df_validation['runtime_cat_171_max'] = df_validation['runtime'].apply(lambda x: 1 if (x >=170) else 0)


    # Budget
    # budget_graph(df_training)
    df_training['budget'] = np.log1p(df_training.budget)
    df_validation['budget'] = np.log1p(df_validation.budget)
    

    # HomePage
    # homePage_graph(df_training)
    df_training['film_that_has_homepage'] = df_training['homepage'].isnull().apply(lambda x: 0 if x == True else 1).copy()
    df_validation['film_that_has_homepage'] = df_validation['homepage'].isnull().apply(lambda x: 0 if x == True else 1).copy()

    
    # Language
    # language_graph(df_training)
    lang = df_training['original_language']
    lang_more_17_samples = [x[0] for x in Counter(pd.DataFrame(lang).stack()).most_common(17)]
    for col in lang_more_17_samples :
        df_training[col] = df_training['original_language'].apply(lambda x: 1 if x == col else 0)
    for col in lang_more_17_samples :
        df_validation[col] = df_validation['original_language'].apply(lambda x: 1 if x == col else 0)

    
    # Genres
    # genres_graph(df_training)
    df_training['genres_names'] = [[y['name'] for y in list(eval(x))] for x in df_training['genres']]

    genres = df_training['genres_names'].sum()
    ctr = Counter(genres)
    genres=[n for n in ctr if ctr[n] > 260]
    genres_list = pd.Series(genres).unique()

    for a in genres_list:
        df_training['genre_'+a] = df_training['genres_names'].apply(lambda x: 1 if a in x else 0)
    df_training = df_training.drop(['genres_names'], axis=1)

    df_validation['genres_names'] = [[y['name'] for y in list(eval(x))] for x in df_validation['genres']]
    for a in genres_list :
        df_validation['genre_'+a] = df_validation['genres_names'].apply(lambda x: 1 if a in x else 0)
    df_validation = df_validation.drop(['genres_names'], axis=1)

    
    # Release data
    # month_graph(df_training)
    df_training=date_features(df_training)
    df_validation=date_features(df_validation)
    

    # Actors
    t = df_training[['movie_id','revenue', 'original_title', 'cast']].copy()
    t['cast'] = [[y['name'] for y in list(eval(x))] for x in t['cast']]
    t['cast'] = t['cast'].apply(lambda x: x[:3])

    names = t['cast'].sum()
    ctr = Counter(names)
    df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'actor', 0:'count'})    
    df_names = df_names[df_names['count'] > 12]
    df_names = df_names.sort_values('count', ascending=False).head(29)
    names_list = list(df_names['actor'])

    df_training['cast_names'] = [[y['name'] for y in list(eval(x))] for x in df_training['cast']]
    df_training['cast_names'] = df_training['cast_names'].apply(lambda x: x[:3])

    for a in names_list :
        df_training['actor_'+a] = df_training['cast_names'].apply(lambda x: 1 if a in x else 0)
    df_training = df_training.drop(['cast_names'], axis=1)

    df_validation['cast_names'] = [[y['name'] for y in list(eval(x))] for x in df_validation['cast']]
    df_validation['cast_names'] = df_validation['cast_names'].apply(lambda x: x[:3])

    for a in names_list :
        df_validation['actor_'+a]=df_validation['cast_names'].apply(lambda x: 1 if a in x else 0)
    df_validation = df_validation.drop(['cast_names'], axis=1)


    # Directors
    t = df_training[['movie_id','revenue', 'original_title', 'crew']].copy()
    t['crew'] = [[y['name'] for y in list(eval(x)) if y['department']=='Directing'] for x in t['crew'] ]
    t['crew'] = t['crew'].apply(lambda x: x[:3])
    
    names = t['crew'].sum()
    ctr = Counter(names)
    df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'director', 0:'count'})       
    df_names = df_names[df_names['count'] > 9]
    df_names = df_names.sort_values('count', ascending=False).head(30)
    
    names_list = list(df_names['director'])
    
    df_training['crew_names'] = [[y['name'] for y in list(eval(x)) if y['department']=='Directing'] for x in df_training['crew'] ]
    df_training['crew_names'] = df_training['crew_names'].apply(lambda x: x[:3])

    for a in names_list:
        df_training['director_'+a] = df_training['crew_names'].apply(lambda x: 1 if a in x else 0)
    df_training = df_training.drop(['crew_names'], axis=1)

    df_validation['crew_names'] = [[y['name'] for y in list(eval(x)) if y['department']=='Directing'] for x in df_validation['crew'] ]
    df_validation['crew_names'] = df_validation['crew_names'].apply(lambda x: x[:3])
    for a in names_list:
        df_validation['director_'+a] = df_validation['crew_names'].apply(lambda x: 1 if a in x else 0)
    df_validation = df_validation.drop(['crew_names'], axis=1)

    
    # Production companies
    t = df_training[['movie_id','revenue', 'original_title', 'production_companies']].copy()
    t['production_companies'] = [[y['name'] for y in list(eval(x))] for x in t['production_companies'] ]
    t['production_companies'] = t['production_companies'].apply(lambda x: x[:3])

    names = t['production_companies'].sum()
    ctr = Counter(names)
    df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'company', 0:'count'})       
    df_names=df_names.sort_values('count', ascending=False)

    df_names = df_names[df_names['count'] > 15]
    names_list = list(df_names['company'])

    df_training['production_companies'] = [[y['name'] for y in list(eval(x))] for x in df_training['production_companies'] ]
    df_training['production_companies'] = df_training['production_companies'].apply(lambda x: x[:3])

    dic={}
    for a in names_list:
        mask = df_training['production_companies'].apply(lambda x: a in x)
        dic[a] = df_training[mask]['revenue'].mean()

    companies_mean_revenue = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'company'})

    names_list = list(companies_mean_revenue.nlargest(40, 'mean_revenue')['company'])

    for a in names_list:
        df_training['production_'+a]=df_training['production_companies'].apply(lambda x: 1 if a in x else 0)
    df_training = df_training.drop(['production_companies'], axis=1)

    df_validation['production_companies'] = [[y['name'] for y in list(eval(x))] for x in df_validation['production_companies']]
    df_validation['production_companies'] = df_validation['production_companies'].apply(lambda x: x[:3])

    for a in names_list:
        df_validation['production_'+a] = df_validation['production_companies'].apply(lambda x: 1 if a in x else 0)
    df_validation = df_validation.drop(['production_companies'], axis=1)


    # Model
    # Create target object and call it y
    train_x = df_training.drop(['movie_id', 'runtime', 'revenue', 'rating'], axis=1).select_dtypes(exclude=['object'])
    train_y = df_training.revenue
    columns = train_x.columns.values.tolist()

    val_x = df_validation.drop(['movie_id', 'runtime', 'revenue', 'rating'], axis=1).select_dtypes(exclude=['object'])
    val_y = df_validation.revenue

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import RandomizedSearchCV

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train_x, train_y)
    pred_y = rf.predict(val_x)

    mse = mean_squared_error(df_validation.revenue, pred_y)
    corr = np.corrcoef(df_validation.revenue, pred_y)[0, 1]
    print(mse)
    print(corr)
    
    summary_dict = {'zid': ['z5110579'], 'MSE': [mse], 'correlation': [corr]}
    summary_df = pd.DataFrame(summary_dict)
    summary_df.to_csv('./z5110579.PART1.summary.csv', sep=',', header=True, index=True)

    output_dict = {}
    output_df = pd.DataFrame()
    output_df['movie_id'] = df_validation['movie_id']
    output_df['predicted_revenue'] = pd.Series(pred_y)
    output_df.to_csv('./z5110579.PART1.output.csv', sep=',', header=True, index=True)



    # --------------- Q2 ----------------
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    from xgboost import XGBClassifier

    df_training = pd.read_csv(sys.argv[1])
    df_validation = pd.read_csv(sys.argv[2])

    df_training['runtime_cat_min_70'] = df_training['runtime'].apply(lambda x: 1 if (x <= 70) else 0)
    df_training['runtime_cat_70_90'] = df_training['runtime'].apply(lambda x: 1 if (x > 70) & (x <= 90) else 0)
    df_training['runtime_cat_90_110'] = df_training['runtime'].apply(lambda x: 1 if (x > 90) & (x <= 110) else 0)
    df_training['runtime_cat_110_130'] = df_training['runtime'].apply(lambda x: 1 if (x > 110) & (x <= 130) else 0)
    df_training['runtime_cat_130_150'] = df_training['runtime'].apply(lambda x: 1 if (x > 130) & (x <= 150) else 0)
    df_training['runtime_cat_150_max'] = df_training['runtime'].apply(lambda x: 1 if (x > 150) else 0)

    df_validation['runtime_cat_min_70'] = df_validation['runtime'].apply(lambda x: 1 if (x <= 70) else 0)
    df_validation['runtime_cat_70_90'] = df_validation['runtime'].apply(lambda x: 1 if (x > 70) & (x <= 90) else 0)
    df_validation['runtime_cat_90_110'] = df_validation['runtime'].apply(lambda x: 1 if (x > 90) & (x <= 110) else 0)
    df_validation['runtime_cat_110_130'] = df_validation['runtime'].apply(lambda x: 1 if (x > 110) & (x <= 130) else 0)
    df_validation['runtime_cat_130_150'] = df_validation['runtime'].apply(lambda x: 1 if (x > 130) & (x <= 150) else 0)
    df_validation['runtime_cat_150_max'] = df_validation['runtime'].apply(lambda x: 1 if (x > 150) else 0)


    # Actors
    t = df_training[['movie_id','revenue', 'original_title', 'cast']].copy()
    t['cast'] = [[y['name'] for y in list(eval(x))] for x in t['cast']]
    t['cast'] = t['cast'].apply(lambda x: x[:3])

    names = t['cast'].sum()
    ctr = Counter(names)
    df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'actor', 0:'count'})    
    df_names = df_names[df_names['count'] > 15]
    df_names = df_names.sort_values('count', ascending=False).head(30)
    actors_list = list(df_names['actor'])

    df_training['cast_names']=[[y['name'] for y in list(eval(x))] for x in df_training['cast']]
    df_training['cast_names'] = df_training['cast_names'].apply(lambda x: x[:3])

    for a in actors_list :
        df_training['actor_'+a]=df_training['cast_names'].apply(lambda x: 1 if a in x else 0)
    df_training = df_training.drop(['cast_names'], axis=1)

    df_validation['cast_names']=[[y['name'] for y in list(eval(x))] for x in df_validation['cast']]
    df_validation['cast_names'] = df_validation['cast_names'].apply(lambda x: x[:3])

    for a in actors_list :
        df_validation['actor_'+a]=df_validation['cast_names'].apply(lambda x: 1 if a in x else 0)
    df_validation = df_validation.drop(['cast_names'], axis=1)


    # Directors
    t = df_training[['movie_id','revenue', 'original_title', 'crew']].copy()
    t['crew'] = [[y['name'] for y in list(eval(x)) if y['department']=='Directing'] for x in t['crew'] ]
    t['crew'] = t['crew'].apply(lambda x: x[:3])
    
    names = t['crew'].sum()
    ctr = Counter(names)
    df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'director', 0:'count'})       
    df_names = df_names[df_names['count'] > 9]
    df_names = df_names.sort_values('count', ascending=False).head(40)
    directors_list = list(df_names['director'])

    df_training['crew_names'] = [[y['name'] for y in list(eval(x)) if y['department']=='Directing'] for x in df_training['crew'] ]
    df_training['crew_names'] = df_training['crew_names'].apply(lambda x: x[:3])

    for a in directors_list:
        df_training['director_'+a] = df_training['crew_names'].apply(lambda x: 1 if a in x else 0)
    df_training = df_training.drop(['crew_names'], axis=1)

    df_validation['crew_names'] = [[y['name'] for y in list(eval(x)) if y['department']=='Directing'] for x in df_validation['crew'] ]
    df_validation['crew_names'] = df_validation['crew_names'].apply(lambda x: x[:3])
    for a in directors_list :
        df_validation['director_'+a] = df_validation['crew_names'].apply(lambda x: 1 if a in x else 0)
    df_validation = df_validation.drop(['crew_names'], axis=1)


    # Production companies
    t = df_training[['movie_id','revenue', 'original_title', 'production_companies']].copy()
    t['production_companies'] = [[y['name'] for y in list(eval(x))] for x in t['production_companies'] ]
    t['production_companies'] = t['production_companies'].apply(lambda x: x[:3])

    names = t['production_companies'].sum()
    ctr = Counter(names)
    df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'company', 0:'count'})       
    df_names = df_names[df_names['count'] > 15]
    df_names = df_names.sort_values('count', ascending=False).head(40)

    companies_list = list(df_names['company'])

    df_training['production_companies'] = [[y['name'] for y in list(eval(x))] for x in df_training['production_companies'] ]
    df_training['production_companies'] = df_training['production_companies'].apply(lambda x: x[:3])
    for a in companies_list:
        df_training['production_'+a]=df_training['production_companies'].apply(lambda x: 1 if a in x else 0)
    df_training = df_training.drop(['production_companies'], axis=1)

    df_validation['production_companies'] = [[y['name'] for y in list(eval(x))] for x in df_validation['production_companies'] ]
    df_validation['production_companies'] = df_validation['production_companies'].apply(lambda x: x[:3])
    for a in companies_list:
        df_validation['production_'+a] = df_validation['production_companies'].apply(lambda x: 1 if a in x else 0)
    df_validation = df_validation.drop(['production_companies'], axis=1)


    # Build the final input features
    train_x = pd.DataFrame()
    train_x = df_training[['runtime_cat_min_70', 'runtime_cat_70_90', 'runtime_cat_90_110', 'runtime_cat_110_130', 'runtime_cat_130_150',\
        'runtime_cat_150_max']].select_dtypes(exclude=['object'])
    for a in companies_list:
        train_x['production_' + a] = df_training['production_' + a]
    for a in actors_list:
        train_x['actor_'+a] = df_training['actor_'+a]
    for a in directors_list:
        train_x['director_'+a] = df_training['director_'+a]

    val_x = pd.DataFrame()
    val_x = df_validation[['runtime_cat_min_70', 'runtime_cat_70_90', 'runtime_cat_90_110', 'runtime_cat_110_130', 'runtime_cat_130_150',\
        'runtime_cat_150_max']].select_dtypes(exclude=['object'])
    for a in companies_list:
        val_x['production_' + a] = df_validation['production_' + a]
    for a in actors_list:
        val_x['actor_'+a] = df_validation['actor_'+a]
    for a in directors_list:
        val_x['director_'+a] = df_validation['director_'+a]

    train_y = df_training.rating
    xgbC = XGBClassifier()
    xgbC.fit(train_x, train_y)
    pred_y = xgbC.predict(val_x)

    acc = accuracy_score(df_validation.rating, pred_y)
    recall = recall_score(df_validation.rating, pred_y, average='macro')
    precision = precision_score(df_validation.rating, pred_y, average='macro')
    print(acc)
    print(recall)
    print(precision)

    # Output the result
    summary_dict = {'zid': ['z5110579'], 'average_precision': [precision], 'average_recall': [recall], 'accuracy': [acc]}
    summary_df = pd.DataFrame(summary_dict)
    summary_df.to_csv('./z5110579.PART2.summary.csv', sep=',', header=True, index=True)

    output_dict = {}
    output_df = pd.DataFrame()
    output_df['movie_id'] = df_validation['movie_id']
    output_df['predicted_rating'] = pd.Series(pred_y)
    output_df.to_csv('./z5110579.PART2.output.csv', sep=',', header=True, index=True)
