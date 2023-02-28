cmake_minimum_required(VERSION 3.10)

project(parserv2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

include_directories(/usr/local/include ..)
link_directories(/usr/local/lib)

aux_source_directory(. PARSER_SOURCES)

