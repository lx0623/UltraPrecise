#!/bin/bash
jdbc_test-8.class jdbc_test.class
java -cp /usr/share/java/mysql-connector-java-8.0.16.jar:. jdbc_test
