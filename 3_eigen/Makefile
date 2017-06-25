.PHONY: all a_check.txt b_check.txt clean 

all : a_check.txt b_check.txt

a_check.txt : amain.py
	python3 $< > $@

b_check.txt : bmain.py
	python3 $< > $@

clean : 
	rm -f *.txt 
	@echo "cleaned up"
