ECHO = echo
FORMAT = pdf
PLOTTER = gnuplot
.PHONY: all a_check.txt plot.$(FORMAT) amain.py bmain.py clean 

all : plot.$(FORMAT) #a_check.txt b_check.txt #c_check.txt

plot.$(FORMAT) : a_check.txt
	$(ECHO) '\
	set term $(FORMAT); \
	set output "$@"; \
	set tics out; \
	set key above ; \
	set xrange [0:10]; \
	set yrange [-1.1:1.1]; \
	set xlabel "x"; \
	set ylabel "y"; \
	plot \
	"$<" index 0 using 1:2 with lines title "y" \
	,"$<" index 0 using 1:3 with lines title "yd" \
	' | $(PLOTTER)

a_check.txt : amain.py
	python3 $< > $@

b_check.txt : bmain.py
	python3 $< > $@

biv_data.txt : bivmain.py
	python3 $< > $@

clean : 
	rm -f a_check.txt b_check.txt biv_data.txt
	@echo "cleaned up"
