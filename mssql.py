import os
from functools import partial

import pymssql
import pandas as pd


def create_conn(host, username, password):
    conn = pymssql.connect(host, username, password)
    cursor = conn.cursor()
    return conn, cursor


def env_variables():
    return {"host": os.environ['MSSQL_HOST'],
            "username": os.environ['MSSQL_USERNAME'],
            "password": os.environ['MSSQL_PASSWORD']}


def create_conn_env():
    args = env_variables()
    return create_conn(**args)


def to_df(query):
    host = os.environ['MSSQL_HOST'] 
    user = os.environ['MSSQL_USERNAME'] 
    password = os.environ['MSSQL_PASSWORD']

    conn = pymssql.connect(host, user, password)
    cursor = conn.cursor()
    cursor.execute(query)
    
    rows = cursor.fetchall()
    column_names = [item[0].lower() for item in cursor.description]
    df = pd.DataFrame(rows, columns=column_names)
    
    conn.close()
    cursor.close()
    return df


def push_query(query, cursor=None):

    if not cursor:
        _, cursor = create_conn_env()

    cursor.execute(query)

    return None


def push_sql(filename, cursor=None):

    if not cursor:
        _, cursor = create_conn_env()

    sql_file = open(filename, 'r')
    sql = sql_file.read()
    sql_file.close()

    sql_commands = sql.split(';')

    for command in sql_commands:
        try:
            cursor.execute(command)
        except Exception as msg:
            print(f"Command skipped: {msg}")