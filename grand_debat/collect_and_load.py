import requests
import os
import pandas as pd


def collect_file(csv_url, filepath):
    response = requests.get(csv_url)
    if response.status_code == 200:
        # Open file and write the content
        with open(filepath, 'wb') as file:
            for chunk in response:
                file.write(chunk)
    return


def collect_all_files(conf, folder):
    for file_name in conf:
        if os.path.exists(os.getcwd() + "/" + folder + "/" + file_name) == False:
            collect_file(conf[file_name]["url"], os.getcwd() + "/" + folder + "/" + file_name)
            print(file_name + " downloaded.")
        else:
            print(file_name + " already exists.")
    return


def load_csv_files(conf, folder):
    df_list = []
    for file_name in conf:
        if os.path.exists(os.getcwd() + "/" + folder + "/" + file_name):
            df = pd.read_csv(os.getcwd() + "/" + folder + "/" + file_name) \
                .drop(conf[file_name]["closed_questions"], axis=1)
            df_list.append(df)
    return pd.concat(df_list, sort=False).fillna("")
