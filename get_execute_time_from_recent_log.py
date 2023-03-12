import re
import argparse
# 因为数据都是 1000000 行
match_str = "{\"plans\":\S*\"inputRows\":10000000\S*\"total\"\S*}"
# print(match_str)
parser = argparse.ArgumentParser(description='log_file_path')
parser.add_argument('log_file_path', type=str,help='log_file_path')

p = re.compile(match_str, re.IGNORECASE)

args = parser.parse_args()
file_name = args.log_file_path

with open("./build/data/log/"+file_name) as fr:
    doc = fr.read()
    for i in p.findall(doc):
        print(i)