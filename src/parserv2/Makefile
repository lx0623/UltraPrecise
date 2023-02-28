
all: scanner.cc parser.cc


parser.cc: parser.yy
	bison -d -o parser.cc parser.yy

scanner.cc: scanner.l
	flex -i --c++ --yylineno -o scanner.cc scanner.l

clean:
	rm -f scanner.cc parser.cc parser.hh

.PHONY : clean