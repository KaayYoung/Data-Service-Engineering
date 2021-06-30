import sklearn
import pandas as pd
import xgboost as xgb
import json
import numpy as np
import sys
from collections import Counter
import bokeh
from bokeh.plotting import figure
from bokeh.io import output_notebook, show
from bokeh.models import LabelSet, ColumnDataSource, HoverTool
from bokeh.palettes import Category20c, Spectral6
from bokeh.transform import cumsum, factor_cmap, jitter


def get_all_genres(set_genres, row):
    print(row)
    if (row is not np.nan):
        genres = set()
        for ele in list(eval(row)):
            
            name = ele['name']
            set_genres.add(name)
            genres.add(name)
        return genres


def runtime_graph(df_training):
    t = df_training[['movie_id', 'revenue', 'original_title']]
    
    hover = HoverTool(tooltips = [
                ('Titre','@original_title'),
                ('Revenue','@revenue'),
                ('id','@movie_id')
            ])

    fig = figure(x_axis_label='Films',
                y_axis_label='Revenue',
                title='Revenue for each Films',
                tools=[hover])

    fig.square(x='movie_id',
            y='revenue',
            source=t)

    show(fig)
    
    print(len(df_training))
    t = df_training[['movie_id','original_title','runtime','revenue']].copy()

    # t.iloc[1335]=t.iloc[1335].replace(np.nan, int(120))
    # t.iloc[2302]=t.iloc[2302].replace(np.nan, int(90))
    # average = t['runtime'].mean()
    # print(average)
    # t['runtime'] = t['runtime'].apply(runtime)
    t['runtime_cat_min_60'] = t['runtime'].apply(lambda x: 1 if (x <= 60) else 0)
    t['runtime_cat_61_80'] = t['runtime'].apply(lambda x: 1 if (x >60) & (x<=80) else 0)
    t['runtime_cat_81_100'] = t['runtime'].apply(lambda x: 1 if (x >80) & (x<=100) else 0)
    t['runtime_cat_101_120'] = t['runtime'].apply(lambda x: 1 if (x >100) & (x<=120) else 0)
    t['runtime_cat_121_140'] = t['runtime'].apply(lambda x: 1 if (x >120) & (x<=140) else 0)
    t['runtime_cat_141_170'] = t['runtime'].apply(lambda x: 1 if (x >140) & (x<=170) else 0)
    t['runtime_cat_171_max'] = t['runtime'].apply(lambda x: 1 if (x >=170) else 0)
    #to count how many samples do we have for a category. We want at at least 15 exemples to categorise a data. 
    # print(Counter(t['runtime_cat_171_max']==1))
    t.loc[t.runtime_cat_min_60 == 1,'runtime_category'] = 'cat_min-60'
    t.loc[t.runtime_cat_61_80 == 1,'runtime_category'] = 'cat_61-80'
    t.loc[t.runtime_cat_81_100 == 1,'runtime_category'] = 'cat_81-100'
    t.loc[t.runtime_cat_101_120 == 1,'runtime_category'] = 'cat_101-120'
    t.loc[t.runtime_cat_121_140 == 1,'runtime_category'] = 'cat_121-140'
    t.loc[t.runtime_cat_141_170 == 1,'runtime_category'] = 'cat_141-170'
    t.loc[t.runtime_cat_171_max == 1,'runtime_category'] = 'cat_171-max'
    cat = t['runtime_category']
    ctr = Counter(cat)
    cat = [x for x in ctr]
    unique_names = pd.Series(cat).unique()

    dic={}
    for a in unique_names:
        mask = t.runtime_category.apply(lambda x: a in x)
        print(mask)
        print(t[mask]['revenue'])
        dic[a] = t[mask]['revenue'].mean()
        
    t = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'runtime_cat'})

    t = t.nlargest(6, 'mean_revenue')
    t['color'] = Category20c[6]

    hover1 = HoverTool(tooltips = [
                ('Runtime_category','@runtime_cat'),
                ('Revenue','@mean_revenue')
            ])

    p = figure(x_range=t.runtime_cat, plot_width=800,plot_height=400, toolbar_location=None, title="Revenue per runtime category", tools=[hover1])
    p.vbar(x='runtime_cat', top='mean_revenue', width=0.9, source=t, legend='runtime_cat',
        line_color='white',fill_color='color')

    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"

    show(p)


def budget_graph(df_training):
    t = df_training[['movie_id','original_title','runtime','revenue','release_date', 'budget']].copy()
    t['revenue'] = np.log1p(t.revenue)

    hover = HoverTool(tooltips = [
            ('Titre','@original_title'),
            ('Revenue','@revenue'),
            ('Budget','@budget')
    ])
    fig = figure(x_axis_label='Budget',
             y_axis_label='Revenue',
             title='log Revenue vs log Budget ',
            tools=[hover])

    fig.square('budget', 'revenue', source=t)

    show(fig)


def homePage_graph(df_training):
    #Plot : Revenue for each film that has homepage or not 

    t = df_training[['revenue','homepage','original_title']].copy()

    t['film_that_has_homepage'] = t['homepage'].isnull().apply(lambda x: str(False) if x==True else str(True))
    t = t.groupby('film_that_has_homepage')['revenue'].mean().reset_index()

    hover1 = HoverTool(tooltips = [
                ('Mean revenue','@revenue'),
    ])

    t['color'] = [Spectral6[1],Spectral6[2]]

    p = figure(x_range=['False','True'], plot_width=600,plot_height=400, toolbar_location=None, title="Revenue for a film that has homepage", tools=[hover1])
    p.vbar(x='film_that_has_homepage', top='revenue', width=0.9, source=t, legend='film_that_has_homepage',
        line_color='white', fill_color='color')

    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = 'top_left'

    show(p)


def language_graph(df_training):
    t = df_training[['revenue','original_language','original_title']].copy()
    lang = t['original_language']
    ctr = Counter(lang).most_common(17)
    lang = [x[0] for x in ctr ]
    unique_names = pd.Series(lang).unique()

    dic={}
    for a in unique_names:
        mask = t.original_language.apply(lambda x: a in x)
        dic[a] = t[mask]['revenue'].mean()

    t = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'langue'})
    t = t.nlargest(12, 'mean_revenue')

    t['color'] = Category20c[12]

    hover1 = HoverTool(tooltips = [
                ('Langue','@langue'),
                ('Revenue','@mean_revenue')
            ])

    p = figure(x_range=t.langue, plot_width=1400,plot_height=400, toolbar_location=None, title="Revenue per original language", tools=[hover1])
    p.vbar(x='langue', top='mean_revenue', width=0.9, source=t, legend='langue',
        line_color='white', fill_color='color')

    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"

    show(p)


def genres_graph(train):
    t = train[['movie_id','revenue', 'original_title', 'genres']].copy()
    print(t['genres'])
    t['genres'] = [[y['name'] for y in list(eval(x))] for x in t['genres']]

    genres = t['genres'].sum()
    ctr = Counter(genres)
    df_genres = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'genre', 0:'count'})       
    df_genres=df_genres.sort_values('count', ascending=False)
    df_genres = df_genres[df_genres['count'] > 1]
    df_genres = df_genres.nlargest(20, 'count')

    genres = list(df_genres['genre'])

    dic={}
    for a in genres:
        mask = t.genres.apply(lambda x: a in x)
        dic[a] = t[mask]['revenue'].mean()

    t = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'genre'})

    t['color'] = Category20c[len(t)]

    hover1 = HoverTool(tooltips = [
                ('Genre','@genre'),
                ('Genre mean revenue','@mean_revenue')
            ])

    p = figure(x_range=t.genre, plot_width=1400,plot_height=400, toolbar_location=None, title="Mean revenue per genre", tools=[hover1])
    p.vbar(x='genre', top='mean_revenue', width=0.9, source=t, legend='genre',
       line_color='white', fill_color='color')

    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"

    show(p)

    t = train[['movie_id','revenue', 'genres']]
    x = [[y['name'] for y in list(eval(x))] for x in t['genres']]
    x = Counter(pd.DataFrame(x).stack())
    x = pd.Series(x)
    data = x.reset_index(name='value').rename(columns={'index':'genre'})
    data['angle'] = data['value']/data['value'].sum() * 2*np.pi
    data['color'] = Category20c[len(x)]

    p = figure(plot_height=350, title="Number of movies per genres", toolbar_location=None,
            tools="hover", tooltips="@genre: @value", x_range=(-0.5, 1.0))
    p.wedge(x=0, y=1, radius=0.4,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend='genre', source=data)

    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None

    show(p)


def actors_graph(train):
    t = train.loc[train['rating']==2]
    t = t[['movie_id','revenue', 'original_title', 'cast']].copy()
    t['cast'] = [[y['name'] for y in list(eval(x))] for x in t['cast']]
    t['cast'] = t['cast'].apply(lambda x: x[:3])

    names = t['cast'].sum()
    ctr = Counter(names)
    df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'actor', 0:'count'})       
    df_names=df_names.sort_values('count', ascending=False)
    df_names = df_names[df_names['count'] > 11]
    
    p = figure(plot_width=1300, plot_height=500, title="Most common actors",
            x_range=df_names['actor'], toolbar_location=None, tooltips=[("Actor", "@actor"), ("Count", "@count")])

    p.vbar(x='actor', top='count', width=1, source=df_names,
        line_color="white" )
    p.y_range.start = 0
    p.x_range.range_padding = 0.05
    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = "Actors name"
    p.xaxis.major_label_orientation = 1.2
    p.outline_line_color = None

    show(p)


def keywords_graph(train):
    t = train.loc[train['rating']==3]
    t = t[['movie_id', 'keywords', 'rating']].copy()
    t['keywords'] = [[y['name'] for y in list(eval(x))] for x in t['keywords']]
    #t['keywords'] = t['keywords'].apply(lambda x: x[:3])
    print(t['keywords'])
    names = t['keywords'].sum()
    ctr = Counter(names)
    df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'keywords', 0:'count'})       
    df_names=df_names.sort_values('count', ascending=False)
    df_names = df_names[df_names['count'] > 30]
    
    p = figure(plot_width=1300, plot_height=500, title="Most common actors",
            x_range=df_names['keywords'], toolbar_location=None, tooltips=[("Keywords", "@keywords"), ("Count", "@count")])

    p.vbar(x='keywords', top='count', width=1, source=df_names,
        line_color="white" )
    p.y_range.start = 0
    p.x_range.range_padding = 0.05
    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = "Keywords name"
    p.xaxis.major_label_orientation = 1.2
    p.outline_line_color = None

    show(p)
    
# def date_process(row):
#     print(row)
#     date = row.split('/')
#     print(date)
#     import sys
#     sys.exit()
#     return date

# feature engeneering : release date 
def date_features(df):
    df[['release_month','release_day','release_year']] = df['release_date'].str.split('-',expand=True).replace(np.nan, 0).astype(int)
    df['release_year'] = df['release_year']
    df.loc[ (df['release_year'] <= 18) & (df['release_year'] < 100), "release_year"] += 2000
    df.loc[ (df['release_year'] > 18)  & (df['release_year'] < 100), "release_year"] += 1900
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_month'] = df['release_date'].dt.month
    # df['release_day'] = df['release_date'].dt.day
    df.drop(columns=['release_day'], inplace=True)
    df.drop(columns=['release_date'], inplace=True)

    return df


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print(len(sys.argv))
        print("Invalid arguments")
        sys.exit()
    print(sys.argv[1])
    print(sys.argv[2])

    df_training = pd.read_csv(sys.argv[1])
    df_validation = pd.read_csv(sys.argv[2])
    
    # training = TRAINING.values[:]
    # validation = VALIDATION.values[:]
    # movie = training[:,0]
    # all_genres = set()
    # df_training['all_genres'] = df_training['genres'].apply(lambda x: get_all_genres(all_genres, x))
    # print(df_training['all_genres'])
    # print(all_genres)


    # train_x = training[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]]
    # train_y = training[:, 13]

    # validation_x = validation[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]]
    # validation_y = validation[:, 13]
    
    # XGBtrain(train_x, train_y, validation_x, validation_y)
    
    # average = df_training['revenue'].mean()

    # print(average)

    # features_graph(df_training)
    # runtime
    # runtime_graph(df_training)

    # feature engeneering : film by runtime category
    # df_training['runtime_cat_min_60'] = df_training['runtime'].apply(lambda x: 1 if (x <=60) else 0)
    # df_training['runtime_cat_61_80'] = df_training['runtime'].apply(lambda x: 1 if (x >60)&(x<=80) else 0)
    # df_training['runtime_cat_81_100'] = df_training['runtime'].apply(lambda x: 1 if (x >80)&(x<=100) else 0)
    # df_training['runtime_cat_101_120'] = df_training['runtime'].apply(lambda x: 1 if (x >100)&(x<=120) else 0)
    # df_training['runtime_cat_121_140'] = df_training['runtime'].apply(lambda x: 1 if (x >120)&(x<=140) else 0)
    # df_training['runtime_cat_141_170'] = df_training['runtime'].apply(lambda x: 1 if (x >140)&(x<=170) else 0)
    # df_training['runtime_cat_171_max'] = df_training['runtime'].apply(lambda x: 1 if (x >=170) else 0)

    # df_validation['runtime_cat_min_60'] = df_validation['runtime'].apply(lambda x: 1 if (x <=60) else 0)
    # df_validation['runtime_cat_61_80'] = df_validation['runtime'].apply(lambda x: 1 if (x >60)&(x<=80) else 0)
    # df_validation['runtime_cat_81_100'] = df_validation['runtime'].apply(lambda x: 1 if (x >80)&(x<=100) else 0)
    # df_validation['runtime_cat_101_120'] = df_validation['runtime'].apply(lambda x: 1 if (x >100)&(x<=120) else 0)
    # df_validation['runtime_cat_121_140'] = df_validation['runtime'].apply(lambda x: 1 if (x >120)&(x<=140) else 0)
    # df_validation['runtime_cat_141_170'] = df_validation['runtime'].apply(lambda x: 1 if (x >140)&(x<=170) else 0)
    # df_validation['runtime_cat_171_max'] = df_validation['runtime'].apply(lambda x: 1 if (x >=170) else 0)


    df_training['runtime_cat_min_70'] = df_training['runtime'].apply(lambda x: 1 if (x <= 70) else 0)
    df_training['runtime_cat_70_90'] = df_training['runtime'].apply(lambda x: 1 if (x > 70)&(x <= 90) else 0)
    df_training['runtime_cat_90_110'] = df_training['runtime'].apply(lambda x: 1 if (x > 90)&(x <= 110) else 0)
    df_training['runtime_cat_110_130'] = df_training['runtime'].apply(lambda x: 1 if (x > 110)&(x <= 130) else 0)
    df_training['runtime_cat_130_150'] = df_training['runtime'].apply(lambda x: 1 if (x > 130)&(x <= 150) else 0)
    df_training['runtime_cat_150_max'] = df_training['runtime'].apply(lambda x: 1 if (x > 150) else 0)

    df_validation['runtime_cat_min_70'] = df_validation['runtime'].apply(lambda x: 1 if (x <= 70) else 0)
    df_validation['runtime_cat_70_90'] = df_validation['runtime'].apply(lambda x: 1 if (x > 70)&(x <= 90) else 0)
    df_validation['runtime_cat_90_110'] = df_validation['runtime'].apply(lambda x: 1 if (x > 90)&(x <= 110) else 0)
    df_validation['runtime_cat_110_130'] = df_validation['runtime'].apply(lambda x: 1 if (x > 110)&(x <= 130) else 0)
    df_validation['runtime_cat_130_150'] = df_validation['runtime'].apply(lambda x: 1 if (x > 130)&(x <= 150) else 0)
    df_validation['runtime_cat_150_max'] = df_validation['runtime'].apply(lambda x: 1 if (x > 150) else 0)


    # Budget
    #budget_graph(df_training)
    df_training['budget'] = np.square(np.log1p(df_training.budget))
    df_validation['budget'] = np.square(np.log1p(df_validation.budget))


    # HomePage
    #homePage_graph(df_training)
    # feature engeneering : Film that has homepage
    df_training['film_that_has_homepage'] = df_training['homepage'].isnull().apply(lambda x: 0 if x == True else 1).copy()
    df_validation['film_that_has_homepage'] = df_validation['homepage'].isnull().apply(lambda x: 0 if x == True else 1).copy()

    
    # Language
    #language_graph(df_training)
    # feature engeneering : one hot encoding for original language that have at least 5 samples
    lang = df_training['original_language']
    lang_more_17_samples = [x[0] for x in Counter(pd.DataFrame(lang).stack()).most_common(17)]

    for col in lang_more_17_samples :
        df_training[col] = df_training['original_language'].apply(lambda x: 1 if x == col else 0)
    for col in lang_more_17_samples :
        df_validation[col] = df_validation['original_language'].apply(lambda x: 1 if x == col else 0)

    
    # Genres
    # genres_graph(df_training)
    
    df_training['genres_names'] = [[y['name'] for y in list(eval(x))] for x in df_training['genres']]

    # genres = train['genres_names'].sum()
    # ctr = Counter(genres)
    # genres=[n for n in ctr if ctr[n] > 249]
    # genres_list = pd.Series(genres).unique()

    genres_list=['Action', 'Adventure', 'Science Fiction', 'Family', 'Fantasy','Animation']
            
    for a in genres_list :
        df_training['genre_'+a]=df_training['genres_names'].apply(lambda x: 1 if a in x else 0)
    df_training = df_training.drop(['genres_names'], axis=1)

    df_validation['genres_names'] = [[y['name'] for y in list(eval(x))] for x in df_validation['genres']]
    for a in genres_list :
        df_validation['genre_'+a]=df_validation['genres_names'].apply(lambda x: 1 if a in x else 0)
    df_validation = df_validation.drop(['genres_names'], axis=1)

    

    # Release data
    df_training=date_features(df_training)
    df_validation=date_features(df_validation)



    # keywords_graph(df_training)

    # Keywords
    t = df_training.loc[df_training['rating'] == 2]
    t = t[['movie_id', 'keywords', 'rating']].copy()
    t['keywords'] = [[y['name'] for y in list(eval(x))] for x in t['keywords']]
    names = t['keywords'].sum()
    ctr = Counter(names)
    df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'keywords', 0:'count'})       
    keywords_df = df_names.sort_values('count', ascending=False).head(5)
    keywords_list = list(keywords_df.nlargest(5, 'count')['keywords'])
    df_training['keyword_names']=[[y['name'] for y in list(eval(x))] for x in df_training['keywords']]
    df_training['keyword_names'] = df_training['keyword_names'].apply(lambda x: x[:5])
    for index, row in keywords_df.iterrows():
        a = row['keywords']
        df_training['keyword_'+a]=df_training['keyword_names'].apply(lambda x: 1 if a in x else 0)
    df_training = df_training.drop(['keyword_names'], axis=1)

    df_validation['keyword_names'] = [[y['name'] for y in list(eval(x))] for x in df_validation['keywords']]
    df_validation['keyword_names'] = df_validation['keyword_names'].apply(lambda x: x[:5])

    for index, row in keywords_df.iterrows():
        a = row['keywords']
        df_validation['keyword_' + a]=df_validation['keyword_names'].apply(lambda x: 1 if a in x else 0)

    df_validation = df_validation.drop(['keyword_names'], axis=1)

    # Keywords
    t = df_training.loc[df_training['rating'] == 3]
    t = t[['movie_id', 'keywords', 'rating']].copy()
    t['keywords'] = [[y['name'] for y in list(eval(x))] for x in t['keywords']]
    names = t['keywords'].sum()
    ctr = Counter(names)
    df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'keywords', 0:'count'})       
    keywords_df = df_names.sort_values('count', ascending=False).head(5)
    keywords_list = list(keywords_df.nlargest(5, 'count')['keywords'])
    df_training['keyword_names']=[[y['name'] for y in list(eval(x))] for x in df_training['keywords']]
    df_training['keyword_names'] = df_training['keyword_names'].apply(lambda x: x[:5])
    for index, row in keywords_df.iterrows():
        a = row['keywords']
        df_training['keyword_'+a]=df_training['keyword_names'].apply(lambda x: 1 if a in x else 0)
    df_training = df_training.drop(['keyword_names'], axis=1)

    df_validation['keyword_names'] = [[y['name'] for y in list(eval(x))] for x in df_validation['keywords']]
    df_validation['keyword_names'] = df_validation['keyword_names'].apply(lambda x: x[:5])

    for index, row in keywords_df.iterrows():
        a = row['keywords']
        df_validation['keyword_' + a]=df_validation['keyword_names'].apply(lambda x: 1 if a in x else 0)

    df_validation = df_validation.drop(['keyword_names'], axis=1)



    # Actors
    t = df_training.loc[df_training['rating']==3]
    t = t[['movie_id','revenue', 'original_title', 'cast']].copy()
    t['cast'] = [[y['name'] for y in list(eval(x))] for x in t['cast']]
    t['cast'] = t['cast'].apply(lambda x: x[:3])

    names = t['cast'].sum()
    ctr = Counter(names)
    df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'actor', 0:'count'})    
    df_names = df_names[df_names['count'] > 12]
    names_list = list(df_names['actor'])

    df_training['cast_names']=[[y['name'] for y in list(eval(x))] for x in df_training['cast']]
    df_training['cast_names'] = df_training['cast_names'].apply(lambda x: x[:3])

    dic={}
    for a in names_list:
        mask = df_training['cast_names'].apply(lambda x: a in x)
        dic[a] = df_training[mask]['revenue'].mean()

    actors_mean_revenue = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'actor'})
    actors_list = list(actors_mean_revenue.nlargest(5, 'mean_revenue')['actor'])
    print(actors_list)
    for a in actors_list :
        df_training['actor_'+a]=df_training['cast_names'].apply(lambda x: 1 if a in x else 0)
    df_training = df_training.drop(['cast_names'], axis=1)

    df_validation['cast_names']=[[y['name'] for y in list(eval(x))] for x in df_validation['cast']]
    df_validation['cast_names'] = df_validation['cast_names'].apply(lambda x: x[:3])

    for a in actors_list :
        df_validation['actor_'+a]=df_validation['cast_names'].apply(lambda x: 1 if a in x else 0)
    df_validation = df_validation.drop(['cast_names'], axis=1)

    final_actors_list = actors_list

    # When rating=2
    t = df_training.loc[df_training['rating']==2]
    t = t[['movie_id','revenue', 'original_title', 'cast']].copy()
    t['cast'] = [[y['name'] for y in list(eval(x))] for x in t['cast']]
    t['cast'] = t['cast'].apply(lambda x: x[:3])

    names = t['cast'].sum()
    ctr = Counter(names)
    df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'actor', 0:'count'})    
    df_names = df_names[df_names['count'] > 9]
    names_list = list(df_names['actor'])

    df_training['cast_names']=[[y['name'] for y in list(eval(x))] for x in df_training['cast']]
    df_training['cast_names'] = df_training['cast_names'].apply(lambda x: x[:3])

    dic={}
    for a in names_list:
        mask = df_training['cast_names'].apply(lambda x: a in x)
        dic[a] = df_training[mask]['revenue'].mean()

    actors_mean_revenue = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'actor'})
    actors_list = list(actors_mean_revenue.nlargest(5, 'mean_revenue')['actor'])
    print(actors_list)
    for a in actors_list :
        df_training['actor_'+a]=df_training['cast_names'].apply(lambda x: 1 if a in x else 0)
    df_training = df_training.drop(['cast_names'], axis=1)

    df_validation['cast_names']=[[y['name'] for y in list(eval(x))] for x in df_validation['cast']]
    df_validation['cast_names'] = df_validation['cast_names'].apply(lambda x: x[:3])

    for a in actors_list :
        df_validation['actor_'+a]=df_validation['cast_names'].apply(lambda x: 1 if a in x else 0)
    df_validation = df_validation.drop(['cast_names'], axis=1)

    final_actors_list = final_actors_list + actors_list
    actors_list = final_actors_list
    print(actors_list)




    # Directors
    t = df_training.loc[df_training['rating']==2]
    t = t[['movie_id','revenue', 'original_title', 'crew']].copy()
    t['crew'] = [[y['name'] for y in list(eval(x)) if y['department']=='Directing'] for x in t['crew'] ]
    t['crew'] = t['crew'].apply(lambda x: x[:3])
    
    names = t['crew'].sum()
    ctr = Counter(names)
    df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'director', 0:'count'})       
    df_names = df_names.sort_values('count', ascending=False)
    df_names = df_names[df_names['count'] > 5]
    
    names_list = list(df_names['director'])
    df_training['crew_names'] = [[y['name'] for y in list(eval(x)) if y['department']=='Directing'] for x in df_training['crew'] ]
    df_training['crew_names'] = df_training['crew_names'].apply(lambda x: x[:3])

    dic={}
    for a in names_list:
        mask = df_training['crew_names'].apply(lambda x: a in x)
        dic[a] = df_training[mask]['revenue'].mean()

    directors_mean_revenue = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'director'})
    directors_list = list(directors_mean_revenue.nlargest(5, 'mean_revenue')['director'])
    final_directors_list = directors_list
    # # train['total_top_actors_revenue']=train['cast_names'].apply(lambda x: sum([1 for i in x if i in names_list]))

    for a in directors_list:
        df_training['director_'+a] = df_training['crew_names'].apply(lambda x: 1 if a in x else 0)
    df_training = df_training.drop(['crew_names'], axis=1)

    df_validation['crew_names'] = [[y['name'] for y in list(eval(x)) if y['department']=='Directing'] for x in df_validation['crew'] ]
    df_validation['crew_names'] = df_validation['crew_names'].apply(lambda x: x[:3])
    for a in directors_list :
        df_validation['director_'+a] = df_validation['crew_names'].apply(lambda x: 1 if a in x else 0)
    df_validation = df_validation.drop(['crew_names'], axis=1)


    # When rating=3
    t = df_training.loc[df_training['rating']==3]
    t = t[['movie_id','revenue', 'original_title', 'crew']].copy()
    t['crew'] = [[y['name'] for y in list(eval(x)) if y['department']=='Directing'] for x in t['crew'] ]
    t['crew'] = t['crew'].apply(lambda x: x[:3])
    
    names = t['crew'].sum()
    ctr = Counter(names)
    df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'director', 0:'count'})       
    df_names = df_names.sort_values('count', ascending=False)
    df_names = df_names[df_names['count'] > 8]
    
    names_list = list(df_names['director'])
    df_training['crew_names'] = [[y['name'] for y in list(eval(x)) if y['department']=='Directing'] for x in df_training['crew'] ]
    df_training['crew_names'] = df_training['crew_names'].apply(lambda x: x[:3])

    dic={}
    for a in names_list:
        mask = df_training['crew_names'].apply(lambda x: a in x)
        dic[a] = df_training[mask]['revenue'].mean()

    directors_mean_revenue = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'director'})
    directors_list = list(directors_mean_revenue.nlargest(5, 'mean_revenue')['director'])
    print(directors_list)
    # # train['total_top_actors_revenue']=train['cast_names'].apply(lambda x: sum([1 for i in x if i in names_list]))

    for a in directors_list:
        df_training['director_'+a] = df_training['crew_names'].apply(lambda x: 1 if a in x else 0)
    df_training = df_training.drop(['crew_names'], axis=1)

    df_validation['crew_names'] = [[y['name'] for y in list(eval(x)) if y['department']=='Directing'] for x in df_validation['crew'] ]
    df_validation['crew_names'] = df_validation['crew_names'].apply(lambda x: x[:3])
    for a in directors_list :
        df_validation['director_'+a] = df_validation['crew_names'].apply(lambda x: 1 if a in x else 0)
    df_validation = df_validation.drop(['crew_names'], axis=1)

    final_directors_list += directors_list
    directors_list = final_directors_list


    # Production companies
    t = df_training.loc[df_training['rating']==2]
    t = t[['movie_id','revenue', 'original_title', 'production_companies']].copy()
    t['production_companies'] = [[y['name'] for y in list(eval(x))] for x in t['production_companies'] ]
    t['production_companies'] = t['production_companies'].apply(lambda x: x[:3])

    names = t['production_companies'].sum()
    ctr = Counter(names)
    df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'company', 0:'count'})       
    df_names=df_names.sort_values('count', ascending=False)

    df_names = df_names[df_names['count'] > 2]
    names_list = list(df_names['company'])

    df_training['production_companies_name'] = [[y['name'] for y in list(eval(x))] for x in df_training['production_companies'] ]
    df_training['production_companies_name'] = df_training['production_companies_name'].apply(lambda x: x[:3])

    dic={}
    for a in names_list:
        mask = df_training['production_companies_name'].apply(lambda x: a in x)
        dic[a] = df_training[mask]['revenue'].mean()

    companies_mean_revenue = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'company'})

    companies_list = list(companies_mean_revenue.nlargest(20, 'mean_revenue')['company'])
    print(companies_list)
    final_companies_list = companies_list
    # train['total_top_companies']=train['production_companies'].apply(lambda x: sum([1 for i in x if i in names_list]))
    for a in companies_list:
        df_training['production_'+a]=df_training['production_companies_name'].apply(lambda x: 1 if a in x else 0)
    df_training = df_training.drop(['production_companies_name'], axis=1)

    df_validation['production_companies_name'] = [[y['name'] for y in list(eval(x))] for x in df_validation['production_companies'] ]
    df_validation['production_companies_name'] = df_validation['production_companies_name'].apply(lambda x: x[:3])
    # test['total_top_companies']=test['production_companies'].apply(lambda x: sum([1 for i in x if i in names_list]))

    for a in companies_list:
        df_validation['production_'+a] = df_validation['production_companies_name'].apply(lambda x: 1 if a in x else 0)
    df_validation = df_validation.drop(['production_companies_name'], axis=1)



    # Production companies
    t = df_training.loc[df_training['rating']==3]
    
    t = t[['movie_id','revenue', 'original_title', 'production_companies']].copy()
    t['production_companies'] = [[y['name'] for y in list(eval(x))] for x in t['production_companies'] ]
    t['production_companies'] = t['production_companies'].apply(lambda x: x[:3])

    names = t['production_companies'].sum()
    ctr = Counter(names)
    df_names = pd.DataFrame.from_dict(ctr, orient='index').reset_index().rename(columns={'index':'company', 0:'count'})       
    # df_names=df_names.sort_values('count', ascending=False).head(15)

    df_names = df_names[df_names['count'] > 15]
    names_list = list(df_names['company'])

    df_training['production_companies'] = [[y['name'] for y in list(eval(x))] for x in df_training['production_companies'] ]
    df_training['production_companies'] = df_training['production_companies'].apply(lambda x: x[:3])

    dic={}
    for a in names_list:
        mask = df_training['production_companies'].apply(lambda x: a in x)
        dic[a] = df_training[mask]['revenue'].mean()

    companies_mean_revenue = pd.DataFrame.from_dict(dic, orient='index', columns=['mean_revenue']).reset_index().rename(columns={'index':'company'})

    companies_list = list(companies_mean_revenue.nlargest(10, 'mean_revenue')['company'])
    print(companies_list)
    # train['total_top_companies']=train['production_companies'].apply(lambda x: sum([1 for i in x if i in names_list]))
    for a in companies_list:
        df_training['production_'+a]=df_training['production_companies'].apply(lambda x: 1 if a in x else 0)
    df_training = df_training.drop(['production_companies'], axis=1)

    df_validation['production_companies'] = [[y['name'] for y in list(eval(x))] for x in df_validation['production_companies'] ]
    df_validation['production_companies'] = df_validation['production_companies'].apply(lambda x: x[:3])
    # test['total_top_companies']=test['production_companies'].apply(lambda x: sum([1 for i in x if i in names_list]))

    for a in companies_list:
        df_validation['production_'+a] = df_validation['production_companies'].apply(lambda x: 1 if a in x else 0)
    df_validation = df_validation.drop(['production_companies'], axis=1)


    final_companies_list += companies_list
    companies_list = final_companies_list


    # tagline
    df_training['exist_tagline'] = df_training['tagline'].apply(lambda x: 1 if x != np.nan else 0)
    df_validation['exist_tagline'] = df_validation['tagline'].apply(lambda x: 1 if x != np.nan else 0)

    # Model
    # Create target object and call it y
    train_x = df_training.drop(['movie_id', 'runtime', 'revenue', 'rating'], axis=1).select_dtypes(exclude=['object'])
    # print(train_x.columns.values.tolist())
    # sys.exit()
    train_x = pd.DataFrame()
    train_x = df_training[['runtime_cat_min_70', 'runtime_cat_70_90', 'runtime_cat_90_110', 'runtime_cat_110_130', 'runtime_cat_130_150',\
        'runtime_cat_150_max']].select_dtypes(exclude=['object'])
    # for a in companies_list:
    #     train_x['production_' + a] = df_training['production_' + a]
    # for a in genres_list:
    #     train_x['genre_'+a]=df_training['genre_'+a]
    
    # for a in keywords_list:
    #     train_x['keyword_'+a] = df_training['keyword_'+a]
    
    for a in actors_list:
        train_x['actor_'+a] = df_training['actor_'+a]
    for a in directors_list:
         train_x['director_'+a] = df_training['director_'+a]
    
    # train_x = df_training[['runtime_cat_min_60', 'runtime_cat_61_80', 'runtime_cat_81_100', 'runtime_cat_101_120', 'runtime_cat_121_140',\
    #     'runtime_cat_141_170', 'runtime_cat_171_max']].select_dtypes(exclude=['object'])
    train_y = df_training.revenue
    columns = train_x.columns.values.tolist()
    print(columns)
    print(len(columns))

    val_x = df_validation.drop(['movie_id', 'runtime', 'revenue', 'rating'], axis=1).select_dtypes(exclude=['object'])
    val_x = pd.DataFrame()
    val_x = df_validation[['runtime_cat_min_70', 'runtime_cat_70_90', 'runtime_cat_90_110', 'runtime_cat_110_130', 'runtime_cat_130_150',\
        'runtime_cat_150_max']].select_dtypes(exclude=['object'])
    # for a in companies_list:
    #     val_x['production_' + a] = df_validation['production_' + a]
    # for a in genres_list :
    #     val_x['genre_'+a]=df_validation['genre_'+a]
    # for a in keywords_list:
    #     val_x['keyword_'+a] = df_validation['keyword_'+a]
    for a in actors_list:
        val_x['actor_'+a] = df_validation['actor_' + a]
    for a in directors_list:
         val_x['director_'+a] = df_validation['director_'+a]
    val_y = df_validation.revenue

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import RandomizedSearchCV
    # n_estimators = [10, 20, 50, 100]
    # max_depth = [10, 20, 30, None]
    # min_samples_split = [2, 5, 10]
    # min_samples_leaf = [1, 2, 4]
    # random_grid = {'n_estimators': n_estimators, 
    #             'max_depth': max_depth,
    #             'min_samples_split': min_samples_split,
    #             'min_samples_leaf': min_samples_leaf}
    # rf = RandomForestRegressor()
    # rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,\
    #      n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)
    
    # rf_random.fit(train_x, train_y)
    # pred_y = rf_random.best_estimator_.predict(val_x)
    # print(np.expm1(pred_y))

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train_x, train_y)
    pred_y = rf.predict(val_x)
    # print(np.expm1(pred_y))

    mse = mean_squared_error(df_validation.revenue, pred_y)
    print(mse)
    import numpy

    corr = numpy.corrcoef(df_validation.revenue, pred_y)[0, 1]
    print(corr)
    # output the result
    summary_dict = {'zid': ['z5110579'], 'MSE': [mse], 'correlation': [corr]}
    summary_df = pd.DataFrame(summary_dict)
    summary_df.to_csv('./z5110579.PART1.summary.csv', sep=',', header=True, index=True)

    output_dict = {}
    output_df = pd.DataFrame()
    output_df['movie_id'] = df_validation['movie_id']
    output_df['predicted_revenue'] = pd.Series(pred_y)
    output_df.to_csv('./z5110579.PART1.output.csv', sep=',', header=True, index=True)
    print(output_df)

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(train_x, train_y)
    pred_y = model.predict(val_x)
    print(numpy.corrcoef(df_validation.revenue, np.expm1(pred_y))[0, 1])
    # xgb_model = xgb.XGBRegressor(learning_rate=0.05, 
    #                         n_estimators=10000,max_depth=4)
    # xgb_model.fit(train_x, train_y)
    # pred_y = xgb_model.predict(val_x)

    # print(np.expm1(pred_y))
    # mse = mean_squared_error(df_validation.revenue, np.expm1(pred_y))
    # print(mse)




    # --------------- Q2 ----------------
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV

    train_y = df_training.rating

    clf = RandomForestClassifier(n_estimators=100, max_depth=4,
                              random_state=24)

    clf.fit(train_x, train_y)
    pred_y = clf.predict(val_x)
    print(pred_y)
    print(accuracy_score(df_validation.rating, pred_y))

    xgbC = XGBClassifier()
    xgbC.fit(train_x, train_y)
    pred_y = xgbC.predict(val_x)
    print(pred_y)
    print(accuracy_score(df_validation.rating, pred_y))

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(solver='liblinear')
    model.fit(train_x, train_y)
    red_y = model.predict(val_x)
    print(pred_y)
    print(accuracy_score(df_validation.rating, pred_y))
