from datetime import datetime
import sqlite3
import pandas as pd
from pathlib import Path


class SQLite:
    def __init__(self, file_name: Path):
        self.connection = sqlite3.connect(file_name)

    def __enter__(self):
        return self.connection.cursor()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.commit()
        self.connection.close()


def create_new_tables(db_file):
    print(f"Creating new tables for database {db_file}")
    sql_create_processed_zip = """CREATE TABLE IF NOT EXISTS processed_zip (
        zip_id INTEGER PRIMARY KEY AUTOINCREMENT,
        zip_path TEXT NOT NULL,
        datetime TEXT NOT NULL);"""
    sql_create_summary = """CREATE TABLE IF NOT EXISTS stoma_summary (
        file_id INTEGER PRIMARY KEY AUTOINCREMENT,
        zip_id INTEGER NOT NULL,
        file_path TEXT NOT NULL,
        unfiltered_num_stomata INTEGER NOT NULL,
        filtered_num_stomata INTEGER NULL,
        FOREIGN KEY(zip_id) REFERENCES processed_zip(zip_id));"""
    sql_create_details = """CREATE TABLE IF NOT EXISTS stoma_details (
        stoma_id INTEGER PRIMARY KEY AUTOINCREMENT,
        summary_file_id INTEGER NOT NULL,
        local_stoma_id INTEGER NOT NULL,
        area INTEGER NOT NULL,
        xmin INTEGER NOT NULL,
        ymin INTEGER NOT NULL,
        xmax INTEGER NOT NULL,
        ymax INTEGER NOT NULL,
        conf_score_per_cent INTEGER NOT NULL,
        FOREIGN KEY(summary_file_id) REFERENCES stoma_summary(file_id));"""
    with SQLite(file_name=db_file) as cursor:
        cursor.execute("DROP TABLE IF EXISTS processed_zip")
        cursor.execute("DROP TABLE IF EXISTS stoma_summary")
        cursor.execute("DROP TABLE IF EXISTS stoma_details")
        cursor.execute(sql_create_processed_zip)
        cursor.execute(sql_create_summary)
        cursor.execute(sql_create_details)
    return


def select_all_processed_zip(db_file):
    print(f"Selecting all processed_zips from db {db_file}...")
    sqlite_select = """SELECT zip_path FROM processed_zip;"""
    with SQLite(file_name=db_file) as cursor:
        cursor.execute(sqlite_select)
        records = cursor.fetchall()
    if len(records) == 0:
        return []
    else:
        return [record[0] for record in records]


def insert_stoma_details_entry(db_file, summary_file_id, stoma_details):
    stoma_details = [(summary_file_id, *entry) for entry in
                     stoma_details]  # append summary_file_id to each entry in stoma_details
    print(f"Adding stoma_details entry to db {db_file}...")
    sqlite_insert = """INSERT INTO stoma_details (summary_file_id, local_stoma_id, area, xmin, ymin, xmax, ymax, conf_score_per_cent) VALUES (?, ?, ?, ?, ?, ?, ?, ?);"""
    with SQLite(file_name=db_file) as cursor:
        cursor.executemany(sqlite_insert, stoma_details)
        last_row_id = cursor.execute("SELECT last_insert_rowid() from stoma_details").fetchone()[0]
    return last_row_id


def insert_stoma_summary_entry(db_file, zip_id, stoma_summary):
    # TODO: might need to apply filters first
    stoma_summary = (zip_id, *stoma_summary)
    print(f"Adding stoma_summary entry to db {db_file}...")
    sqlite_insert = """INSERT INTO stoma_summary (zip_id, file_path, unfiltered_num_stomata) VALUES (?, ?, ?);"""
    with SQLite(file_name=db_file) as cursor:
        cursor.execute(sqlite_insert, stoma_summary)
        last_row_id = cursor.execute("SELECT last_insert_rowid() from stoma_summary").fetchone()[0]
    return last_row_id


def insert_processed_zip_entry(db_file, zip_file_path):
    print(f"Adding processed_zip entry to db {db_file}...")
    sqlite_insert = """INSERT INTO processed_zip (zip_path, datetime) VALUES (?, ?);"""
    with SQLite(file_name=db_file) as cursor:
        cursor.execute(sqlite_insert, (zip_file_path, datetime.utcnow()))
        last_row_id = cursor.execute("SELECT last_insert_rowid() from processed_zip").fetchone()[0]
    return last_row_id


def select_stoma_details(db_file):
    print(f"Selecting all stoma_details from db {db_file}...")
    sqlite_select = """SELECT * FROM stoma_details;"""
    with SQLite(file_name=db_file) as cursor:
        cursor.execute(sqlite_select)
        records = cursor.fetchall()
    if len(records) == 0:
        return []
    else:
        df = pd.DataFrame(records,
                          columns=['stoma_id', 'summary_file_id', 'local_stoma_id', 'area', 'xmin', 'ymin', 'xmax',
                                   'ymax', 'conf_score_per_cent'])
        return df


def select_stoma_summary(db_file):
    print(f"Selecting all stoma_summary from db {db_file}...")
    sqlite_select = """SELECT file_id, file_path, filtered_num_stomata FROM stoma_summary;"""
    with SQLite(file_name=db_file) as cursor:
        cursor.execute(sqlite_select)
        records = cursor.fetchall()
    if len(records) == 0:
        return []
    else:
        df = pd.DataFrame(records,
                          columns=['file_id', 'file_path', 'filtered_num_stoma'])
        return df


def select_stoma_summary_details(db_file):
    print(f"Selecting all stoma_summary and details from db {db_file}...")
    sqlite_select = """
    SELECT 
    stoma_summary.file_id AS file_id,
    stoma_summary.file_path AS file_path,
    stoma_details.local_stoma_id AS local_stoma_id,
    stoma_details.area AS area,
    stoma_details.xmin AS xmin,
    stoma_details.ymin AS ymin,
    stoma_details.xmax AS xmax,
    stoma_details.ymax AS ymax,
    stoma_details.conf_score_per_cent AS conf_score
    FROM stoma_details
    LEFT JOIN stoma_summary on stoma_summary.file_id = summary_file_id;"""
    with SQLite(file_name=db_file) as cursor:
        cursor.execute(sqlite_select)
        records = cursor.fetchall()
    if len(records) == 0:
        print(f"No files found without filtered_num_stomata")
        return []
    else:
        df = pd.DataFrame(records,
                          columns=['file_id', 'file_path', 'local_stoma_id', 'area', 'xmin', 'ymin', 'xmax', 'ymax',
                                   'conf_score_per_cent'])
        return df


def update_stoma_summary(db_file, stoma_summary):
    # TODO: need to throw an error, if the file_id is not there
    print(f"Updating stoma_summary from db {db_file}...")
    sqlite_update = """UPDATE stoma_summary SET filtered_num_stomata = ? WHERE file_id = ?"""
    with SQLite(file_name=db_file) as cursor:
        cursor.executemany(sqlite_update, stoma_summary)
    return


def select_joined_stoma_summary(db_file):
    print(f"Selecting all stoma summaries from db {db_file} in easy readable format")
    sqlite_select = """
    SELECT 
    stoma_summary.file_id AS file_id,
    stoma_summary.file_path AS file_path,
    stoma_summary.unfiltered_num_stomata AS unfiltered_num_stomata,
    stoma_summary.filtered_num_stomata AS filtered_num_stomata,
    processed_zip.zip_path as zip_path,
    processed_zip.datetime as uploaded_datetime
    FROM stoma_summary
    LEFT JOIN processed_zip on processed_zip.zip_id = stoma_summary.zip_id;"""

    with SQLite(file_name=db_file) as cursor:
        cursor.execute(sqlite_select)
        records = cursor.fetchall()
    if len(records) == 0:
        return []
    else:
        df = pd.DataFrame(records,
                          columns=['file_id', 'file_path', 'unfiltered_num_stomata', 'filtered_num_stomata', 'zip_path',
                                   'uploaded_datetime'])
        return df


def select_joined_stoma_details(db_file):
    print(f"Selecting all stoma details from db {db_file} in easy readable format")
    sqlite_select = """
    SELECT 
    processed_zip.zip_path as zip_path,
    stoma_summary.file_path AS file_path,
    stoma_details.local_stoma_id AS local_stoma_id,
    stoma_details.area AS area,
    stoma_details.xmin AS xmin,
    stoma_details.ymin AS ymin,
    stoma_details.xmax AS xmax,
    stoma_details.ymax AS ymax,
    stoma_details.conf_score_per_cent AS conf_score
    FROM stoma_details
    LEFT JOIN stoma_summary on stoma_summary.file_id = summary_file_id
    LEFT JOIN processed_zip on processed_zip.zip_id = stoma_summary.zip_id;"""

    with SQLite(file_name=db_file) as cursor:
        cursor.execute(sqlite_select)
        records = cursor.fetchall()
    if len(records) == 0:
        return []
    else:
        df = pd.DataFrame(records,
                          columns=['zip_path', 'file_path', 'local_stoma_id', 'area', 'xmin', 'ymin', 'xmax', 'ymax', 'conf_score_per_cent'])
        return df
