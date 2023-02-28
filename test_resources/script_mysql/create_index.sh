. env.rc

cat mysql_index.sql | mysql -h127.0.0.1 -u$USER -p$PASSWORD -D $DB_NAME > /dev/null
