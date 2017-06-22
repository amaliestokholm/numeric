.PHONY: clean

all : check 

output_a.txt : a.py
	@echo '(a) What is your name?';
	python3 $^ > $@ 

output_b.txt : b.py
	@echo '(b) What is your name?';
	python3 $^ > $@

check: output_a.txt correct_a.txt output_b.txt correct_b.txt
	@echo "Checking part a ..."
	@diff --brief output_a.txt correct_a.txt
	@echo "Part a is okay"
	@echo "Checking part b ..."
	@diff --brief output_b.txt correct_b.txt
	@echo "Part b is okay"

correct_a.txt:
	echo "What's your name? \nHello $(shell whoami)" > $@

correct_b.txt:
	echo "What's your name? \nHello $(shell whoami)!" > $@


clean:
	$(RM) *.txt
