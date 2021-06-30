from flask import Flask, g, request, send_file
from flask_restx import fields, Api, Resource, reqparse, inputs
from contextlib import closing
from datetime import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import pandas as pd
import requests
import sqlite3
import json
import numpy as np

app = Flask(__name__)
api = Api(app,
          defalut="TV Shows",
          title="TV Shows DataSet",
          description="An example to show how to import and manipulate tvshows.")

schedule_model = api.model('Schedule', {
    'time': fields.String,
    'days': fields.List(fields.String)
})
rating_model = api.model('Rating', {
    'average':fields.String
})

country_model = api.model('Country', {
    'name': fields.String,
    'code': fields.String,
    'timezone': fields.String
})
network_model = api.model('Network', {
    'id': fields.Integer,
    'name': fields.String,
    'country': fields.Nested(country_model)    
})

show_model = api.model('TV_SHOW', {
    'tvmaze-id':fields.Integer,
    'name':fields.String,
    'type':fields.String,
    'language':fields.String,
    'genres':fields.List(fields.String),
    'status':fields.String,
    'runtime':fields.Integer,
    'premiered':fields.String,
    'officialSite':fields.String,
    'schedule':fields.Nested(schedule_model),
    'rating':fields.Nested(rating_model),
    'weight':fields.Integer,
    'network':fields.Nested(network_model),
    'summary':fields.String
})

input_name_model = api.model('Name', {'name':fields.String})
orderBy_model = api.model('OrderBy', {
    'orderBy':fields.String, 
    'page':fields.Integer,
    'page_size':fields.Integer,
    'filter':fields.String    
})

db_features = ['tvmaze_id', 'name' ,'type' ,'language' ,'genres' ,'status' ,'runtime',\
    'premiered', 'officialSite', 'schedule', 'rating', 'weight', 'network', 'summary']
orderBy_all = ['id', 'name', 'runtime', 'premiered', 'rating-average']
filter_list = ['tvmaze_id', 'id', 'last-update', 'name', 'type', 'language', 'genres', 'status', 'runtime',\
    'premiered', 'officialSite', 'schedule', 'rating', 'weight', 'network', 'summary']
parser = reqparse.RequestParser()
parser.add_argument('name', type=str, default='string: name', required=True)


@app.before_request
def before_request():
    g.db = sqlite3.connect("z5110579.db")
    with closing(sqlite3.connect("z5110579.db")) as db:
        db.cursor().execute(''' 
            CREATE TABLE if not exists tv
            (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            info String NOT NULL )'''
        )
        db.commit()


@app.after_request
def after_request(response):
    g.db.close()
    return response


@api.route('/tv_shows/import')
@api.param('name', 'TV Show name')
class tvShows(Resource):
    @api.response(404, 'The tvshow was not found')
    @api.response(400, 'Validation Error')
    @api.response(201, 'The TV Show imported Successfully')
    @api.doc(description='Import a tvshow to db')
    # @api.expect(input_name_model)
    def post(self):
        tvshow_input = parser.parse_args()
        # tvshow_input = request.json
        if 'name' not in tvshow_input:
            return {"message": "Missing Name"}, 400
        
        tv_name = tvshow_input['name']
        
        r = requests.get("http://api.tvmaze.com/search/shows?q=" + tv_name)
        
        if (len(r.json()) == 0):
            return {"message": "No matched tv with this tv_name"}, 404
        else:
            df_data = pd.read_sql('select * from tv', g.db)
            df_data['tvmaze-id'] = df_data['info'].apply(lambda x: json.loads(x)['tvmaze_id'])
            target_tv = r.json()[0]['show']
            find_exist = (df_data['tvmaze-id'] == target_tv["id"]).any()
            if (find_exist):
                return {"message": "The tvshow with this tvmaze-id %d already in the database"%(target_tv["id"])}, 400

            target_tv["tvmaze_id"] = target_tv["id"]
            for key in list(target_tv): 
                if key not in db_features:
                    del target_tv[key]
            
            curTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            target_tv['last-update'] = curTime

            info = json.dumps(target_tv)
            info = info.replace("'", "''")
            
            sql = "insert into tv VALUES (NULL, '%s')" % (info)
            cursor = g.db.cursor()
            cursor.execute(sql)
            g.db.commit()

            # Get the id from the last inserted row 
            rowId = cursor.lastrowid
            sql = "select id from tv where rowid = %d" %(rowId)
            cursor.execute(sql)
            resId = cursor.fetchone()[0]
            cursor.close()

            link = {
                "self": {
                    "href":"http://127.0.0.1:5000/tv-shows/%d"%(resId)
                }
             }
            
            return {"id": resId, "last-update": curTime, "tvmaze-id": target_tv["tvmaze_id"], "_links": link}, 201


def checkExist(id):
    cursor = g.db.cursor()
    sql = "select count(*) from tv where id = %d" %id
    cursor.execute(sql)
    data = cursor.fetchone()[0]
    cursor.close()
    return data


def get_links(id):
     # Read data from db
    df_data = pd.read_sql('select * from tv', g.db)

    # Get the index where the row['id'] == id
    target = df_data.iloc[:, [0, -1]][df_data[df_data.T.index[0]] == id]
    cur_index = target.index[0]
    _links = {
        "self":{
            "href":"http://127.0.0.1:5000/tv-shows/%d" % (id)
        }
    }
    # If the current index is not the first one
    if (cur_index > 0):
        _links["prev"] = {"href": "http://127.0.0.1:5000/tv-shows/%d" %(df_data.iloc[cur_index - 1, :]["id"]) }
    # If the current index is not the last one
    if (cur_index < df_data.shape[0] - 1):
        _links["next"] = {"href": "http://127.0.0.1:5000/tv-shows/%d" %(df_data.iloc[cur_index + 1, :]["id"]) }
    return _links


@api.route('/tv_shows/<int:id>')
@api.param('id', 'The TV Show identifier')
class tvShow(Resource):
    @api.response(404, 'The TV Show not found')
    @api.response(200, 'Successful')
    @api.doc(description="Get a tvshow by id")
    def get(self, id):
        if (checkExist(id) == 0):
            api.abort(404, "The tv show with id %d doesn't exist"%(id))
        else:
            cursor = g.db.cursor()
            sql = "select * from tv where id = %d" % id
            cursor.execute(sql)
            res = json.loads(cursor.fetchone()[1])
            # Put id in the json
            res["id"] = id
            res["_links"] = get_links(id)
            
            return res, 200

    @api.response(404, 'The TV Show not found')
    @api.response(200, 'Successful')
    @api.doc(description='delete a tvshow by id')
    def delete(self, id):
        if (checkExist(id) == 0):
            api.abort(404, "The tv show with id %d doesn't exist"%(id))
        else:
            show_json = request.json
            cursor = g.db.cursor()
            sql = "delete from tv where id = %d" % id
            cursor.execute(sql)
            g.db.commit()
            return {"message": "The tv show with id %d was removed from the database!"%(id), "id": id}, 200

    @api.response(404, 'The TV Show not found')
    @api.response(200, 'Successful')
    @api.doc(description='Update a tvshow by id')
    @api.expect(show_model, validate=True)
    def patch(self, id):
        if (checkExist(id) == 0):
            api.abort(404, "The tv show with id %d doesn't exist"%(id))
        else:
            data = request.json
            cursor = g.db.cursor()
            sql = "select * from tv where id = %d" %id
            cursor.execute(sql)
            record = json.loads(cursor.fetchone()[1])
            for k in data:
                record[k] = data[k]
            
            info = json.dumps(record)
            info = info.replace("'", "''")
            sql = "update tv set info='%s' where id = %d" % (info, id)
            cursor.execute(sql)
            g.db.commit()

            res = {
                "id": id,
                "last-update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "_links": get_links(id)
            }
            return res, 200


orderParser = reqparse.RequestParser()
orderParser.add_argument('orderBy', type=str, default="+id")
orderParser.add_argument('page', type=int, default=1)
orderParser.add_argument('page_size', type=int, default=100)
orderParser.add_argument('filter', type=str, default="id,name")

@api.route('/tv_show')
class orderBy(Resource):
    @api.response(400, 'Validation Error')
    @api.response(200, 'Successful')
    @api.doc(parser=orderParser,
        description="order tvshows by some parameters")
    def get(self):
        data = orderParser.parse_args()
        
        # Below is to check invalid inputs
        sortByValues = []
        sortByAscending = []
        for s in data['orderBy'].split(','):
            s = s.strip()
            sign = s[0:1]
            if (sign == '-'):
                sortByAscending.append(False)
            elif (sign == '+'):
                sortByAscending.append(True)
            else:
                api.abort(400, "The OrderBy parameter doesn't have +/- sign")
            ele = s[1:len(s)]
            if (ele in orderBy_all):
                sortByValues.append(ele)
            else:
                api.abort(400, "The orderBy parameter is invalid")
        filters = data['filter'].split(",")
        for i in range(len(filters)):
            filters[i] = filters[i].strip()
            if filters[i] not in filter_list:
                api.abort(400, "invalid filters input")
        
        page = data['page']
        page_size = data['page_size']
        cursor = g.db.cursor()
        cursor.execute('select count(*) from tv')
        count = cursor.fetchone()[0]
        if ((page - 1) * page_size >= count):
            api.abort(400, "page or page_size is invalid")
        
        # Get all records from db
        df_data = pd.read_sql('select * from tv', g.db)

        combined_df = pd.DataFrame()
        for index, row in df_data.iterrows():
            info = row["info"]
            json_info = json.loads(info)
            
            toDf = {}
            for i in orderBy_all:
                if i == "rating-average":
                    toDf[i] = json_info["rating"]["average"]
                elif i == "id":
                    toDf[i] = row["id"]
                else:
                    toDf[i] = json_info[i]
            
            filters_df = {}
            filters_df["id"] = row["id"]
            for i in filters:

                if i != "id":
                    filters_df[i] = json_info[i]
            toDf["info"] = json.dumps(filters_df)
            combined_df = combined_df.append(pd.DataFrame(data=toDf, index=[0]))

        # Sort the dataframe
        sorted_df = combined_df.sort_values(by=sortByValues, ascending=tuple(sortByAscending))

        # Apply pagination on the sorted dataframe
        res_df = sorted_df[(page - 1) * page_size: page * page_size]
        
        # change the string format to json
        tvshows = res_df['info'].tolist()
        res_tvshows = []
        for i in tvshows:
            res_tvshows.append(json.loads(i))

        # Add links
        _links = {}
        if page > 1 :
            link_prev = {"href": "http://127.0.0.1:5000/tv-shows?page=%d&page_size=%d&filter=%s" %(page - 1, page_size, data['filter'])}
            _links["prev"] = link_prev
        link_self = {"href": "http://127.0.0.1:5000/tv-shows?page=%d&page_size=%d&filter=%s" %(page, page_size, data['filter'])}
        _links["self"] = link_self
        page_nums = int(count / page_size) + 1 

        if page < page_nums:
            link_next = {"href": "http://127.0.0.1:5000/tv-shows?page=%d&page_size=%d&filter=%s" %(page, page_size, data['filter'])}
            _links["next"] = link_next

        res = {
            "page": page,
            "page-size": page_size,
            "tvshows": res_tvshows,
            "_links": _links
        }
        
        return res, 200


statisticsParser = reqparse.RequestParser()
statisticsParser.add_argument('format', type=str)
statisticsParser.add_argument('by', type=str)

sta_by_list = ['language', 'genres', 'status', 'type']


def statistics_genres(row):
    genres = json.loads(row)['genres']
    genres.sort()
    s = ''
    for ele in genres:
        s += ele + '|'
    return s[0:len(s) - 1]


# display the percentage along with the number
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct


@api.route('/tv_shows/statistcs')
class statistics(Resource):
    @api.response(400, 'Validation Error')
    @api.response(200, 'Successful')
    @api.doc(parser=statisticsParser,
        description="order tvshows by some parameters")
    def get(self):
        data = statisticsParser.parse_args()
        sta_format = data['format']
        if (sta_format != "json" and sta_format != "image"):
            api.abort(400, "format parameter is invalid")
        sta_by = data['by']
        if (sta_by not in sta_by_list):
            api.abort(400, "by paramter is invalid")
        
        cursor = g.db.cursor()

        # Calculate the total number of records
        cursor.execute('select count(*) from tv')
        count = cursor.fetchone()[0]

        # Read data from db to dataframe
        df_data = pd.read_sql('select * from tv', g.db)

        # Get the current time
        curTime = datetime.now()
        # Calculate time diff for all rows
        df_data['timeDiff'] = df_data['info'].apply(lambda x: (curTime - datetime.strptime(json.loads(x)['last-update'], "%Y-%m-%d %H:%M:%S")).days)
        # Calculate the number of records within 24h
        num_24 = int((df_data['timeDiff'] == 0).sum())
        
        # Create a new column named by "by" parameter(language, genres, status, type)
        if (sta_by == 'genres'):
            df_data[sta_by] = df_data['info'].apply(statistics_genres)
        else:
            df_data[sta_by] = df_data['info'].apply(lambda x: json.loads(x)[sta_by])
        res_df = df_data.groupby(df_data[sta_by])[sta_by].count()
        sum_by = res_df.sum()
        
        # Calculate the percentage
        res_df = res_df.apply(lambda x: x/sum_by)

        if (sta_format == 'json'):
            values = {}
            for i, v in res_df.items():
                values[i] = v
            res = {
                "total": count,
                "total-updated": num_24,
                "values": values
            }
            return res, 200
        else:
            # num_info = pd.DataFrame({'non-update': count - num_24, 'last-update': num_24}, index=["update"])
            num_info = pd.DataFrame({'update': [count - num_24, num_24]}, index=["non-update", 'last-update'])
            fig, axes = plt.subplots(2, 1, figsize=(13,13))
            fig.suptitle("--- Q6 --- Percentage & Info about number of tvshows")
            resPlot = res_df.plot.pie(ax=axes[0], autopct='%.2f%%', radius=1.25)
            # axes[0].set_title('Percentage of tvshows', fontsize=14)
            # axes[0].axis('equal')

            infoPlot = num_info['update'].plot.pie(ax=axes[1], autopct=make_autopct([count-num_24, num_24]), radius=1.25)
            # axes[1].set_title('Info about the number of tvshows', fontsize=14)
            # axes[1].axis('equal')
            plt.savefig("{}.png".format("z5110579"))

            return send_file(r"z5110579.png", mimetype='image/png', cache_timeout=0)


if __name__ == '__main__':
    app.run(debug=True)
