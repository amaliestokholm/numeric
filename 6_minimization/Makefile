FORMAT = pdf
PLOTTER = gnuplot
.PHONY: all a_check.txt b_check.txt biv_data.txt amain.py bmain.py bivmain.py clean 

all : a_check.txt b_check.txt #c_check.txt

plot.$(FORMAT) : biv_data.txt
	$(RM) $(SCRIPT)
	$(ECHO) '\
		set term $(FORMAT); \
		set output "$@"; \
		set title "Fit to radioactive decay"; \
		set tics out; \
		set key above ; \
		set xrange [0:10]; \
		set xlabel "t"; \
		set ylabel "Activity"; \
		plot \
		"$<" using 1:2  with lines title "Fit" \
		, "$<" using 3:4 with points title "Experiments" \
		, "$<" using 5 with errorbars \
		' | tee $(SCRIPT) | $(PLOTTER)

a_check.txt : amain.py
	python3 $< > $@

b_check.txt : bmain.py
	python3 $< > $@

biv_data.txt : bivmain.py
	python3 $< > $@

clean : 
	rm -f a_check.txt b_check.txt biv_data.txt
	@echo "cleaned up"
