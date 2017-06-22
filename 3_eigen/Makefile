.PHONY: all acheck.txt clean 

all : acheck.txt

acheck.txt : amain.py
	python3 $< > $@

clean : 
	rm -f *.txt 
	@echo "cleaned up"
