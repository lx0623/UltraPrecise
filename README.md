# Fast-APA
Fast-APA: An Arbitrary-Precision Arithmetic Framework for Databases on GPU

## Start Fast-APA
- Make related dir\
```mkdir build``` 
- Compile system\
```cd build```\
```make -j```
- Init system schema\
```./rateup --initialize --datadir=data```
- Run server\
```./rateup <--port=3306> <--datadir=data> ```
- Run mysql cli\
```mysql -h:: <-urateup> <-p>```

The log files are stored in build/data/log

Details of the experiment are stored in the append_expr_info folder