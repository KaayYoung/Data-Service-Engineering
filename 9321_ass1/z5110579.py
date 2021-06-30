import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import numpy as np
import math

studentid = os.path.basename(sys.modules[__name__].__file__)


def get_distance(latitude, longitude, wuhan_lat=114.3055, wuhan_long=30.5928):
    R = 6373
    pi = 3.1416
    C = math.sin(latitude*pi/180) * math.sin(wuhan_lat*pi/180) * math.cos((longitude - wuhan_long)*pi/180) +\
        math.cos(latitude*pi/180) * math.cos(wuhan_lat*pi/180)
    return R * math.acos(C)


def match_cont_country(cont):
    cont.loc[71 ,'Country'] = 'North Korea'
    cont.loc[72, 'Country'] = 'South Korea'
    cont.loc[125, 'Country'] = 'North Macedonia'
    cont.loc[7, 'Country'] = 'Cabo Verde'
    cont.loc[85, 'Country'] = 'Russia'
    cont.loc[12, 'Country'] = 'Democratic Republic of the Congo'
    cont.loc[11, 'Country'] = 'Republic of the Congo'
    cont.loc[47, 'Country'] = 'Eswatini'
    cont.loc[4, 'Country'] = 'Burkina Faso'
    cont.loc[109, 'Country'] = 'Czech Republic'
    cont.loc[59, 'Country'] = 'Myanmar'
    cont.loc[167, 'Country'] = 'United States of America'
    
    
def helper_q6(group):
    cities = group.iloc[0, 2]    
    not_none = [[x.get('City'), x.get('Population')] for x in cities if x.get('Population') != None]
    res = pd.DataFrame(not_none, columns=['City', 'Population'])
    return res


def helper_q7(group):
    s = group.iloc[0, 1]
    country = group.iloc[0, 0]
    res = np.unique([a.get('City') for a in s])
    pairs = [[x, country] for x in res]
    return pd.DataFrame(pairs, columns=['City', 'Country'])


def helper_q9(row):
    return [np.mean(row['Covid_19_Economic_exposure_index_Ex_aid_and_FDI']), np.mean(row['Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import']),\
         np.mean(row['Foreign direct investment, net inflows percent of GDP']), np.mean(row['Foreign direct investment'])]


def log(question, output_df, other):
    print("--------------- {}----------------".format(question))

    if other is not None:
        print(question, other)
    if output_df is not None:
        df = output_df.head(5).copy(True)
        for c in df.columns:
            df[c] = df[c].apply(lambda a: a[:20] if isinstance(a, str) else a)

        df.columns = [a[:10] + "..." for a in df.columns]
        print(df.to_string())


def question_1(exposure, countries):
    """
    :param exposure: the path for the exposure.csv file
    :param countries: the path for the Countries.csv file
    :return: df1
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    #################################################
    # Your code goes here ...
    df1 = pd.read_csv(exposure, sep=";")
    df1 = df1.dropna(subset=['country'])

    df2 = pd.read_csv(countries)

    df1.loc[45, 'country'] = 'Democratic Republic of the Congo'
    df1.loc[56, 'country'] = 'North Korea'
    df1.loc[73, 'country'] = 'Laos'
    df1.loc[88, 'country'] = 'Republic of the Congo'
    df1.loc[97, 'country'] = 'Brunei'
    df1.loc[108, 'country'] = 'Vietnam'
    df1.loc[150, 'country'] = 'Ivory Coast'
    df1.loc[159, 'country'] = 'Moldova'
    df1.loc[161, 'country'] = 'Russia'
    df1.loc[179, 'country'] = 'South Korea'
    df2['Country'] = df2['Country'].replace('United States', 'United States of America')
    df2['Country'] = df2['Country'].replace('Macedonia', 'North Macedonia')
    df2['Country'] = df2['Country'].replace('Palestinian Territory', 'Palestine')
    df2['Country'] = df2['Country'].replace('Cape Verde', 'Cabo Verde')
    df2['Country'] = df2['Country'].replace('Swaziland', 'Eswatini')
    
    df1 = pd.merge(left=df1, right=df2, on=None, left_on='country', right_on='Country')
    
    df1.set_index('Country', inplace=True)
    del df1['country']
    df1.sort_index(inplace=True)
    #################################################
    log("QUESTION 1", output_df=df1, other=df1.shape)
    return df1


def helper_q2(row):
    arr = row['Cities']
    lat = []
    lon = []
    for ele in arr:
        lat.append(ele.get('Latitude'))
        lon.append(ele.get('Longitude'))
    return np.mean(lat), np.mean(lon)


def question_2(df1):
    """
    :param df1: the dataframe created in question 1
    :return: df2
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df2 = df1
    df2.loc[:, 'Cities'] = df2.loc[:, 'Cities'].apply(lambda x: [json.loads(ele) for ele in x.split("|||")])
    df2[['avg_latitude', 'avg_longitude']] = df2.apply(helper_q2, axis=1, result_type="expand")
    #################################################
    log("QUESTION 2", output_df=df2[["avg_latitude", "avg_longitude"]], other=df2.shape)
    return df2


def question_3(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df3
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    #################################################
    # Your code goes here ...
    df3 = df2[['avg_latitude', 'avg_longitude']]
    df3['distance_to_Wuhan'] = df3[['avg_latitude', 'avg_longitude']].\
        apply(lambda x: get_distance(x['avg_latitude'], x['avg_longitude']), axis=1)
    df3.sort_values(by=['distance_to_Wuhan'], inplace=True)
    # #################################################
    log("QUESTION 3", output_df=df3[['distance_to_Wuhan']], other=df3.shape)
    return df3


def question_4(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :param continents: the path for the Countries-Continents.csv file
    :return: df4
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    # Read csv
    cont = pd.read_csv(continents)
    match_cont_country(cont)
    cont = cont.append(pd.Series({'Country': 'Palestine', 'Continent': 'Asia'}), ignore_index=True)
    
    # Extract target columns and merge
    info = df2[['Covid_19_Economic_exposure_index']]
    info.reset_index(inplace=True)

    info = pd.merge(left=info, right=cont, on=None, left_on='Country', right_on='Country')
    info['Covid_19_Economic_exposure_index'] = info['Covid_19_Economic_exposure_index'].apply(lambda x: x.replace(',', '.')).replace('x', '-1').astype(float)
    df4 = info[info['Covid_19_Economic_exposure_index'] != -1.0].groupby('Continent')['Covid_19_Economic_exposure_index'].mean()
    
    # Transform the Series Object to DataFrame
    dict_df4 = {'Continent': df4.index, 'Covid_19_Economic_exposure_index': df4.values}
    df4 = pd.DataFrame(dict_df4)

    # Set Continent as index and sort by Covid_19_exposure
    df4.set_index('Continent', inplace=True)
    df4.sort_values(by=['Covid_19_Economic_exposure_index'], inplace=True)
    #################################################

    log("QUESTION 4", output_df=df4, other=df4.shape)
    return df4


def question_5(df2):
    """
    :param df2: the dataframe created in question 2
    :return: cities_lst
            Data Type: list
            Please read the assignment specs to know how to create the output dataframe
    """
    #################################################
    # Your code goes here ...
    info = df2[['Income classification according to WB', 'Net_ODA_received_perc_of_GNI', 'Foreign direct investment']]
    net_oda = info[['Income classification according to WB']]
    net_oda.loc[:, 'Net_ODA_received_perc_of_GNI'] = info.loc[:, 'Net_ODA_received_perc_of_GNI'].apply(lambda x: x.replace(',', '.')).replace('No data', np.nan).astype(float)
    net_oda.dropna(axis=0, subset=['Net_ODA_received_perc_of_GNI'])

    foreign = info[['Income classification according to WB']]
    foreign.loc[:, 'Foreign direct investment'] = info.loc[:, 'Foreign direct investment'].apply(lambda x: x.replace(',', '.')).replace('x', np.nan).astype(float)
    foreign.dropna(axis=0, subset=['Foreign direct investment'])

    df5 = pd.DataFrame()
    df5['avg_Net_ODA_received_perc_of_GNI'] = net_oda.groupby('Income classification according to WB')['Net_ODA_received_perc_of_GNI'].mean()
    df5['avg_Foreign direct investment'] = foreign.groupby('Income classification according to WB')['Foreign direct investment'].mean()
    #################################################

    log("QUESTION 5", output_df=df5, other=df5.shape)
    return df5


def question_6(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df6
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    
    #################################################
    # Your code goes here ...
    info = df2[['Income classification according to WB', 'Cities']]
    info.reset_index(inplace=True)
    info = info[info['Income classification according to WB'] == 'LIC']

    df6 = pd.DataFrame()
    df6[['City', 'Population']] = info.groupby('Country', group_keys=False).apply(helper_q6)
    df6.sort_values(by=['Population'], ascending=False, inplace=True)
    cities_lst = df6.head(5)['City'].values.tolist()
    #################################################

    log("QUESTION 6", output_df=None, other=cities_lst)
    return cities_lst


def question_7(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df7
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    
    #################################################
    # Your code goes here ...
    pre = df2[['Cities']]
    pre.reset_index(inplace=True)
    info = pre.groupby('Country', group_keys=False).apply(helper_q7)
    info = info[info.groupby('City')['Country'].transform('count') > 1]

    # Transform series object to list
    df7 = info.groupby('City')['Country'].apply(lambda x: list(x))

    dict_df7 = {'City': df7.index, 'Countries': df7.values}
    df7 = pd.DataFrame(dict_df7)
    df7.set_index('City', inplace=True)
    #################################################

    log("QUESTION 7", output_df=df7, other=df7.shape)
    return df7


def question_8(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :param continents: the path for the Countries-Continents.csv file
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    cont = pd.read_csv(continents)

    # If there is no South America Country, then don't need to change continents data
    match_cont_country(cont)
    cont = cont.append(pd.Series({'Country': 'Palestine', 'Continent': 'Asia'}), ignore_index=True)

    info = df2[['Cities']]
    info.reset_index(inplace=True)

    info = pd.merge(left=info, right=cont, on=None, left_on='Country', right_on='Country')

    # Begin to calculate Total population of the world
    pop_info = info[['Country', 'Continent']]
    pop_info.loc[:,'country_pop'] = info['Cities'].apply(lambda x: np.sum([ele.get('Population') for ele\
         in x if ele.get('Population') is not None]))
    total_pop = pop_info['country_pop'].sum()

    # Get the South America Countries
    SA = pop_info[pop_info['Continent'] == 'South America']
    del SA['Continent']
    SA.set_index('Country', inplace=True)

    SA = SA.apply(lambda x: x / total_pop)
    SA_plot = SA.plot(kind='bar', figsize=(10, 10))
    SA_plot.set_title("Q8")
    SA_plot.set_xlabel("Country")
    SA_plot.set_ylabel("Population Percentage of the World")
    #################################################

    plt.savefig("{}-Q11.png".format(studentid))


def question_9(df2):
    """
    :param df2: the dataframe created in question 2
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    props = ['Covid_19_Economic_exposure_index_Ex_aid_and_FDI', 'Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import',\
        'Foreign direct investment, net inflows percent of GDP', 'Foreign direct investment']
    info = df2[['Covid_19_Economic_exposure_index_Ex_aid_and_FDI', 'Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import', 'Foreign direct investment, net inflows percent of GDP', 'Foreign direct investment']]

    info = info.apply(lambda x: x.str.replace(',', '.')).replace('x', np.nan).astype(float)
    info['Income classification according to WB'] = df2['Income classification according to WB']
    series = info.groupby('Income classification according to WB').apply(helper_q9)

    p1 = []
    p2 = []
    p3 = []
    p4 = []
    for i in range(3):
        cur = series[i]
        p1.append(cur[0])
        p2.append(cur[1])
        p3.append(cur[2])
        p4.append(cur[3])
    res = pd.DataFrame([p1, p2, p3, p4], columns=['HIC', 'LIC', 'MIC'])
    res.set_index(pd.Index(['Covid_19_Economic_exposure_index_Ex_aid_and_FDI', 'Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import',\
        'Foreign direct investment, net inflows percent of GDP', 'Foreign direct investment']), inplace=True)
    
    q9 = res.plot(kind='bar', figsize=(10, 10))
    q9.set_title("Q9")
    q9.set_xlabel("Metrics")
    q9.set_ylabel("Units")
    #################################################

    plt.savefig("{}-Q12.png".format(studentid))


def question_10(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :return: nothing, but saves the figure on the disk
    :param continents: the path for the Countries-Continents.csv file
    """

    #################################################
    # Your code goes here ...
    cont = pd.read_csv(continents)
    match_cont_country(cont)
    cont = cont.append(pd.Series({'Country': 'Palestine', 'Continent': 'Asia'}), ignore_index=True)
    
    # Country pop is used to describe the size of point
    info = df2[['Cities', 'avg_latitude', 'avg_longitude']]
    info.loc[:, 'country_pop'] = info.loc[:, 'Cities'].apply(lambda x: np.sum([ele.get('Population')\
         for ele in x if ele.get('Population') != None]))
    del info['Cities']
    info.reset_index(inplace=True)

    info = pd.merge(left=info, right=cont, on=None, left_on='Country', right_on='Country')
    continents = info.groupby('Continent')
    
    # Begin plotting
    import matplotlib.pyplot as plt
    color_dict = {'South America':'green', 'North America':'gray', 'Africa':'yellow', 'Oceania':'blue', 'Asia': 'red', 'Europe':'pink'}

    fig, ax = plt.subplots()
    fig.suptitle("Countries Location", fontsize=18)
    fig.set_figheight(8)
    fig.set_figwidth(9)
    for name, group in continents:
        color = color_dict[name]
        ax.scatter(group.avg_longitude, group.avg_latitude, color=color, s=group.country_pop/800000, label=name)
        ax.set_xlabel('avg_longitude', fontsize=14)
        ax.set_ylabel('avg_latitude', fontsize=14)
        ax.legend()
    #################################################

    plt.savefig("{}-Q13.png".format(studentid))


if __name__ == "__main__":
    df1 = question_1("exposure.csv", "Countries.csv")
    df2 = question_2(df1.copy(True))
    df3 = question_3(df2.copy(True))
    df4 = question_4(df2.copy(True), "Countries-Continents.csv")
    df5 = question_5(df2.copy(True))
    lst = question_6(df2.copy(True))
    df7 = question_7(df2.copy(True))
    question_8(df2.copy(True), "Countries-Continents.csv")
    question_9(df2.copy(True))
    question_10(df2.copy(True), "Countries-Continents.csv")
