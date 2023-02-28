#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os, stat

# you can change the bleow configurations
executorPath = "/home/server/data_for_pie/script/orc-csv-import"
sourcePath = "/data/tpch/data/scale_5/csv/org"
soucePostfix = ".tbl"
targetPath = "/data/tpch/data/scale_5/csv/aries_double"
targetPostfix = ""
sqlFile= "/data/tpch/data/scale_5/script_aries/dss.ddl.double"
outShellFile= "/data/tpch/data/scale_5/script_aries/import_aries_double_4_tpch_5.sh"
#mode 1: to ORC file, 2: to Aries file, <0: show data
importMode = "2"
delimiter = "|"

# ========================================
delimiterCh = "--delimiter=$delimiter"
Mode = "--mode=$importMode"

with open(outShellFile, 'w+') as f:
  f.write("#!/bin/bash\n\n")
  f.write("executorPath=" + executorPath + "\n")
  f.write("sourcePath=" + sourcePath + "\n")
  f.write("soucePostfix=" + soucePostfix + "\n")
  f.write("targetPath=" + targetPath + "\n")
  f.write("targetPostfix=" + targetPostfix + "\n")
  f.write("delimiter='" + delimiter + "'\n")
  f.write("#importMode 1: to ORC file, 2: to Aries file, <0: show data\n")
  f.write("importMode='" + importMode + "'\n")
  f.write("mkdir -p " + targetPath + "\n\n")

def myStrip(line):
  line = line.strip().lower()
  res = ""
  havespace = False
  for c in line:
    if c != ' ':
      havespace = False
      res += c
    elif havespace == False:
      res += c
      havespace = True
  return res.strip()

with open (sqlFile, 'r') as f:
  content = f.readlines()

tableName = ""
schema = ""
columnId = 0
decimalLists = []
for line in content:
  line = myStrip(line)
  if line.startswith("--"):
    continue
  if line.startswith("/*"):
    continue
  
  if len(line) == 0:
    continue

  if line.startswith("create table"):
    tableName = line.split(' ')[2]
    if tableName[0] == '`':
      tableName = tableName[1:-1]

  #start
  if line == "(" or line.endswith(" ("):
    schema = ""
    columnId = 0
    decimalLists = []
  #skip last line
  elif line.startswith("primary key"):
    continue
  #skip KEY `abc` (`abc`),
  elif line.startswith("key "):
    continue
  #end
  elif line == ");" or line.startswith(") "):
    #skip dbgen_version
    if tableName == "dbgen_version":
      continue
    schema += ">"
    result = ["$executorPath", delimiterCh, Mode, "$schema", "$sourcePath/$tableName$soucePostfix", "$targetPath/$tableName$targetPostfix"]

    with open(outShellFile, 'a+') as f:
      f.write("\n#" + tableName + "\n")
      f.write("tableName=" + tableName + "\n")
      f.write("schema=\"" + schema + "\"\n")
      f.write(" ".join(result) + "\n")
    #print(" ".join(result))

  else:
    if line[-1] == ',':
      line = line[:-1]
    keys = line.split(' ')
    if len(keys) >=2:
      columnName = keys[0]
      if columnName[0] == '`':
        columnName = columnName[1:-1]
      columnType = keys[1]
      if columnType == "integer":
        columnType = "int"
      elif columnType.startswith("int("):
        columnType = "int"
      elif columnType.startswith("bigint("):
        columnType = "long"
      columnId += 1

      NotOrNul = 'nul';
      if len(keys) == 2:
        NotOrNul = 'nul';
      elif len(keys) >= 4:
        if keys[2] == 'not':
          NotOrNul = 'not';
        elif keys[2] == "default":
          NotOrNul = 'nul';

      if len(schema) == 0:
        schema += "struct<" + columnName + ":" + NotOrNul + ":" + columnType
      else:
        schema += "," + columnName + ":" + NotOrNul + ":" + columnType

os.chmod(outShellFile, stat.S_IRWXU)
print("generate shell file complete: " + outShellFile)


