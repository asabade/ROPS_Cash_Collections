import os
from functools import partial

import jaydebeapi
import pandas as pd

from pathlib import Path

import random

def create_conn(host, database, port, username, password):
    connection_string=f'jdbc:netezza://{host}:{port}/{database}'
    url = f'{connection_string}:user={username};password={password}'

    conn = jaydebeapi.connect("org.netezza.Driver",
                              connection_string, 
                              {'user': username, 'password': password},
                              jars = "/work/drivers/nz/nzjdbc3.jar")

    cursor = conn.cursor()
    return conn, cursor


def env_variables():
    return {"host": os.environ['NZ_HOST'],
            "database": os.environ['NZ_DATABASE'],
            "port": os.environ['NZ_PORT'],
            "username": os.environ['NZ_USERNAME'],
            "password": os.environ['NZ_PASSWORD']}


def create_conn_env():
    args = env_variables()
    return create_conn(**args)


def to_df(query, cursor=None):
    if not cursor:
        _, cursor = create_conn_env()

    cursor.execute(query)
    columns = [desc[0].lower() for desc in cursor.description]

    return pd.DataFrame(cursor.fetchall(), columns=columns)

def push_df(df, table_name, cursor=None):

    tmp_base_path = Path('/data/tmp')
    tmp_base_path.mkdir(parents=True, exist_ok=True)
    file_path = tmp_base_path / f'file_{random.sample(range(1000), 1)[0]}.csv'
    df.to_csv(file_path, index=False)
    print('File path:', file_path)
    database = os.environ['NZ_DATABASE']
    username = os.environ['NZ_USERNAME']
    table_column_types = f'''({", ".join([f'{x} varchar(255)' for x in df.columns])})'''
    create_query = f'''CREATE TABLE IF NOT EXISTS {database}.{username}.{table_name} {table_column_types}'''  #AS SELECT * FROM EXTERNAL '{file_path}' {table_column_types} '''
    insert_prefix = f'''INSERT INTO  {database}.{username}.{table_name} SELECT * FROM EXTERNAL '{file_path}' '''
    query_suffix = f''' USING(DELIM ',' DATESTYLE 'YMD' DATEDELIM '-' REMOTESOURCE 'JDBC' MAXERRORS 5000000000 skiprows 1 fillrecord)
        '''
    if not cursor:
        _, cursor = create_conn_env()

    try:
        cursor.execute(create_query)
        push_query = insert_prefix + query_suffix
        cursor.execute(push_query)
    except Exception as e:
        print(f'Insert into datalab_iff_1.{username}.{table_name} failed.')
        print(e)


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
        log.info(command)
        try:
            cursor.execute(command)
            log.info('Command completed')
        except Exception as msg:
            log.info(f"Command skipped: {msg}")
