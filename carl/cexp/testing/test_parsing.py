import os
import json
import glob
import cexp.env.datatools.data_parser as parser

if __name__ == '__main__':

    jsonfile = 'database/sample_benchmark.json'

    with open(jsonfile, 'r') as f:
        ._json = json.loads(f.read())
    parserd_exp_dict = parser.parse_exp_vec(_json['envs'])




    # We get all the jsons and test the files
    glob.glob('*.json')
