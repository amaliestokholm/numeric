ECHO = /bin/echo

.PHONY: all check.txt clean 

all : check.txt

check.txt : main.py
	python3 $< > $@

clean : 
	rm -f *.txt 
	@echo "cleaned up"
