from flask import Flask, g, render_template, request, make_response, jsonify
from flask_restx import fields, Api, Resource
from contextlib import closing
import requests
import sqlite3
import time
import json

app = Flask(__name__)
api = Api(app)


def init_db():
    with closing(sqlite3.connect("z5110579.db")) as db:
        db.cursor().execute(''' 
            CREATE TABLE if not exists tv
            (id INTEGER PRIMARY KEY AUTOINCREMENT, 
            info String NOT NULL )'''
        )
        db.commit()

@app.before_request
def before_request():
    g.db = sqlite3.connect("z5110579.db")
    with closing(sqlite3.connect("z5110579.db")) as db:
        db.cursor().execute(''' 
            CREATE TABLE if not exists tv
            (
            info String NOT NULL )'''
        )
        db.commit()


@app.after_request
def after_request(response):
    g.db.close()
    return response

# @api.route('/tv_shows/import')
# class tvShows(Resource):

@api.route('/tv_shows/import', methods=['POST'])
def import_TV_show():
    tv_name = request.args.get('name')
    # tv_name = request.form['name']
    r = requests.get("http://api.tvmaze.com/search/shows?q=" + tv_name)
    target_tv = r.json()[1]['show']
    
    print("before insert into db")
    cursor = g.db.cursor()
    cursor.execute(''' SELECT * FROM tv''')
    g.db.commit()
    rowId, updateTime = store_TV_show(json.dumps(target_tv))


    # response = make_response("<html>Created<html>", 201)

    # Q1 should display :201 Created
    r = {"id": rowId, "last-update": updateTime, "tvmaze-id": target_tv["id"], "_links": target_tv["_links"]["self"]}
    # return redirect(url_for('import'))
    return make_response(jsonify(r), 201)


def store_TV_show(info):
    info = info.replace("'", "''")
    # info = jsonify(info)
    # print(info)
    # conn = sqlite3.connect("z5110579.db")
    # cursor = conn.cursor()
    sql = "insert into tv VALUES ('%s')" % (info)
    cursor = g.db.cursor()
    cursor.execute(sql)
    rowId = cursor.lastrowid
    g.db.commit()
    curTime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    cursor.close()

    return rowId, curTime


@api.route('/tv_shows/<id>', methods=['GET'])
def retrieve_tv(id):
    cursor = g.db.cursor()
    sql = "select * from tv where rowid = %d" % int(id)
    cursor.execute(sql)
    
    return make_response(cursor.fetchone()[0], 200)


@api.route('/tv_shows/<id>', methods=['DELETE'])
def delete_tv(id):
    id = int(id)
    cursor = g.db.cursor()
    sql = "select count(*) from tv where rowid = %d" %id
    cursor.execute(sql)
    data = cursor.fetchone()[0]
    print(data)
    if (data == 0):
        res = {"message": "The tv show with id %d was not existed in the database"%(id)}
        res = make_response(res, 404)
    else:
        sql = "delete from tv where rowid = %d" % id
        cursor.execute(sql)
        g.db.commit()
        res = {"message": "The tv show with id %d was removed from the database!"%(id), "id": id}
        res = make_response(res, 200)
    return res


@api.route('/tv_shows/<id>', methods=['PATCH'])
def update_tv(id):
    id = int(id)
    data = request.form
    print(data)
    cursor = g.db.cursor()
    sql = "update from tv where rowid = %d" % id
    cursor.execute(sql)


if __name__ == '__main__':
    app.run(debug=True)
