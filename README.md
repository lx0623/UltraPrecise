# Fast-APA

Fast-APA: An Arbitrary-Precision Arithmetic Framework for Databases on GPU

## How to install

/home/dbuser

├─Fast-APA  
├─heavydb  
├─RateupDB   
├─MonetDB  
├─postgresql-14.4  
├─cockroach  
└─h2  

### Install Fast-APA

- Make related dir  
  ```cd Fast-APA```  
  ```mkdir build```   
- Compile system  
  ```cd Fast-APA```  
  ```cd build```  
  ```make -j```
- Init system schema  
  ```cd Fast-APA```  
  ```cd build```  
  ```./rateup --initialize --datadir=data```  
- Run server  
  ```cd Fast-APA```  
  ```cd build```  
  ```./rateup <--port=3306> <--datadir=data> ```  
- Run mysql cli  
  ```mysql -h:: <-urateup> <-p>```  

The log files are stored in build/data/log

### Install Heavy.AI

- Get source code  
  ``git clone git@github.com:heavyai/heavydb.git``

- Install prebuilt dependencies.  
  ```cd heavydb```  
  ```./scripts/mapd-deps-prebuilt.sh```  
  ```source /usr/local/mapd-deps/mapd-deps.sh```  
  ```export LD_LIBRARY_PATH="/usr/local/mapd-deps/lib/:$LD_LIBRARY_PATH"```  
  ```cd scripts```  
  ```CFLAGS="-DSQLITE_ENABLE_COLUMN_METADATA=1" ./configure```  
  ```make```  
  ```sudo make install```  
  ```source /usr/local/mapd-deps/mapd-deps.sh```  

- Compile system  
  ```cd heavydb```  
  ```mkdir build```  
  ```cd build```  
  ```cmake ..```  
  ```make -j```  

- Run server  
  ```mkdir data```  
  ```./bin/initheavy data```  
  ```./bin/heavydb```  

- Run heavysql cli  
  ```./bin/heavysql -p HyperInteractive```  

### Install RateupDB  

- Consistent with Fast-APA  
```git clone git@github.com:lx0623/DB.git```   
### Install MonetDB

- Get source code  
  ```hg clone http://dev.monetdb.org/hg/MonetDB/```  

- Compile System  
  ```cd MonetDB```  
  ```mkdir build```  
  ```cd build```  
  ```cmake -DCMAKE_INSTALL_PREFIX=~/install_monetdb ~/MonetDB/```  
  ```cmake --build .```  
  ```cmake --build . --target install```  
  ```export PATH=$PATH:~/monetdb/install_monetdb/bin```  

- Run Server  
  ```monetdbd create ~/my-dbfarm```  
  ```monetdbd start ~/my-dbfarm```  

- Run mclient  
  ```monetdb create expr```  
  ```monetdb release expr```  
  ```mclient --timer=performance -d expr```  

### Install PostgreSQL

- Get source code  
  ```wget https://ftp.postgresql.org/pub/source/v14.4/postgresql-14.4.tar.gz```  
  ```tar -zxvf postgresql-14.4.tar.gz```  

- Compile System  
  ```cd postgresql-14.4/```  
  ```mkdir postgresql```  
  ```./configure --prefix=/home/dbuser/postgresql-14.4/postgresql```  
  ```make && make install```

- Adduser  
  ```sudo chown -R postgres /home/dbuser/postgresql-14.4/```  
  ```adduser postgres```  
  ```su postgres```  

- Init system schema  
  ```/home/dbuser/postgresql-14.4/postgresql/bin/initdb -D /home/dbuser/postgresql-14.4/postgresql/data```  

- Run server  
  ```/home/dbuser/postgresql-14.4/postgresql/bin/pg_ctl -D /home/dbuser/postgresql-14.4/postgresql/data/ -l logfile start```  

- Run pgsq  
  ```/home/dbuser/postgresql-14.4/postgresql/bin/createdb expr```  
  ```/home/dbuser/postgresql-14.4/postgresql/bin/psql expr```  

### Install CockroachDB

- Get source code  
  ```git clone https://github.com/cockroachdb/cockroach```  

- Install Bazelisk  
  ```https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/  bazelisk-linux-amd64```  
  ```chmod +x bazelisk-linux-amd64```  
  ```sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel```  

- Install CockroachDB  
  ```cd cockroach```  
  ```./dev doctor```  
  ```sudo apt-get install libresolv-wrapper```  
  ```bazel build pkg/cmd/cockroach-short```  

- Run server  
  ```cd ~/cockroach/_bazel/bin/pkg/cmd/cockroach-short/cockroach-short_```  
  ```./cockroach-short start-single-node --insecure```  

- Run client  
  ```cd ~/cockroach/_bazel/bin/pkg/cmd/cockroach-short/cockroach-short_```  
  ```./cockroach-short  sql --insecure```

### Install H2

- Get source code   
  ```https://github.com/h2database/h2database/releases/download/version-2.1.214/h2-2022-06-13.zip```   

- Install H2   
  ```unzip unzip h2-2022-06-13.zip```   

- Run clinet  
  ```cd h2```  
  ```java -cp h2-*.jar org.h2.tools.Shell```  

## How to generate data

├─append_expr_info  
│      ├─gen_data  
│      │      ├─1.dec_representation  
│      │      │      ├─Q1  
│      │      │      └─Q2  
│      │      ├─2.opt_expr  
│      │      │      ├─alignmnet_scheduling  
│      │      │      ├─constant_construction  
│      │      │      └─constant_pre_calculation  
│      │      ├─3.multi_thread  
│      │      │      ├─agg  
│      │      │      └─expr_evaluation  
│      │      └─4.synthesized_workload  
│      │                  ├─RSA  
│      │                  ├─Sin  
│      │                  └─TPC-H_Q1  
│      └─expr_query   
└─Others

To generate the data, simply go to the appropriate folder and run the ```gen.sh``` script

Here is an example of generating Query 1 data:   

```cd append_expr_info```  
```cd gen_data```  
```cd 1.dec_representation```  
```bash gen.sh```  

Generating a ```lineitem.tbl``` for TPC-H and its extensions is a special case. You need to put a ```lineitem.tbl``` under the ```4.synthesized_workload/TPC-H_Q1```.

## How to load data to the database
### Fast-APA and RateupDB
```load data infile 'YourPath/xxx.tbl' into table tableName fields terminated by '|';``` 
### Heavy.AI
Since heavy.AI will have restrictions about whitelisting, we use the ```./bin/heavydb --allowed-import-paths '["/"]' --allowed-export-path '["/"]'``` command to start the service.

```COPY tableName FROM 'YourPath/xxx.tbl' WITH (delimiter= '|',header='false');```  
### MonetDB
```COPY INTO sys.tableName from 'YourPath/xxx.tbl' DELIMITERS '|' ;```
### PostgreSQL
```COPY tableName(col1,col2,xxx) FROM 'YourPath/xxx.tbl' delimiter '|' ;```
### CockroachDB
If you have the same path as the installation above, then you need to put the data into the ```/home/dbuser/cockroach/_bazel/bin/pkg/cmd/cockroach-short/cockroach-short_/cockroach-data/extern```  file path in advance.
In addition you can check the console output about the ```external I/O path``` when you start the service.

```IMPORT INTO tableName(col1,col2,xxx) CSV  DATA("nodelocal://self/xxx.tbl") with delimiter = e'|';```
### H2
The tbl to csv script ```tbl2csv.py``` is stored in the ```gen_data``` directory, and it is worth noting that you have to add the column names first.
```create table tableName( col1 xxx, col2 xxx, xxxxxx) AS SELECT * FROM CSVREAD('YourPath/xxx.csv');```


## How to get the execution time

We use special statements in each database to eliminate the effect of print cost reduction on the experimental results (TPC-H Q1 experiments and aggregation experiments do not consider this case).

Specifically, for Fast-APA, RateupDB, Heavy.AI, we count the time by checking the `logs`.

For MonetDB, we have added the ```TRACE``` statement before the Query command.

For PostgreSQL, CockroachDB and H2, we use the ```Explain Analyze``` statement.

Fast-APA and RateupDB logs are stored in ```build/data/log``` as ```rateup.xxx.xxx.log.INFO.xxx.xxx```.

The execution time is profiled as a JSON data like ```{"plans": [{"compileKernel": xxx,"plan": [{xxxxxx}]}],"total": xxx}}```

It can be extracted from the log file using ```python3 get_execute_time_from_recent_log.py rateup.xxx.xxx.log.INFO.xxx.xxx ```

Heavy.AI's log files are stored in the ```build/storage/log``` folder as ```heavydb.INFO.xxx-xxx```,  and require the addition of the compile option ```./bin/heavydb   --enable-debug-timer=true```.

More details of the experiment are stored in the ```append_expr_info``` folder.