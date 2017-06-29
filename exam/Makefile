ECHO = echo
FORMAT = pdf
PLOTTER = gnuplot
.PHONY: all clean 

all : nerror.$(FORMAT)

nerror.$(FORMAT) : check.txt 
	$(ECHO) '\
	set term $(FORMAT); \
	set output "$@"; \
	set title "Error as a function of sample size"; \
	set tics out; \
	set key above ; \
	set xlabel "Sample size [N]"; \
	set ylabel "Error"; \
	plot \
	"$<" using 1:2 with lines title "MC plain - exact" \
	, "$<" using 1:3 with lines title "MC quasi - exact" \
	, "$<" using 1:4 with lines ls 1 dashtype 2 title "Estimated error on plain" \
	, "$<" using 1:5 with lines ls 2 dashtype 2 title "Estimated error on quasi" \
	' | $(PLOTTER)

check.txt : main.py
	python3 $< > $@

clean : 
	rm -f check.txt nerror.$(FORMAT)
	@echo "cleaned up"
