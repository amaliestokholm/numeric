.PHONY: all a_check.txt b_check.txt amain.py bmain.py clean 

all : a_check.txt b_check.txt

a_check.txt : amain.py
	python3 $< > $@

b_check.txt : bmain.py
	python3 $< > $@

clean : 
	rm -f a_check.txt b_check.txt 
	@echo "cleaned up"
