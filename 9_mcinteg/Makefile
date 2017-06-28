ECHO = echo
FORMAT = pdf
PLOTTER = gnuplot
.PHONY: all a_check.txt aerr.txt plot.$(FORMAT) amain.py clean 

all : a_check.txt plot.$(FORMAT)

plot.$(FORMAT) : aerr.txt
	$(ECHO) '\
	set term $(FORMAT); \
	set output "$@"; \
	set tics out; \
	set key above ; \
	set logscale xy ; \
	set xlabel "Error"; \
	set ylabel "N"; \
	plot \
	"$<" index 0 using 1:2 with lines \
	' | $(PLOTTER)

a_check.txt : amain.py
	python3 $< > $@ 2> aerr.txt

clean : 
	rm -f a_check.txt aerr.txt plot.$(FORMAT)
	@echo "cleaned up"
