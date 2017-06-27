FORMAT = pdf
PLOTTER = gnuplot
.PHONY: all a_check.txt b_check.txt biv_data.txt amain.py bmain.py bivmain.py clean 

all : a_check.txt b_check.txt plot.$(FORMAT) 

plot.$(FORMAT) : biv_data.txt data.txt
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
		"$(word 1, $^)" using 1:2  with lines title "Fit" \
		, "$(word 2, $^)" using 1:2 with points title "Experiments" \
		, "$(word 2, $^)  using 3 with errorbars \
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
