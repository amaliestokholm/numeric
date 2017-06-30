ECHO = echo
FORMAT = pdf
PLOTTER = gnuplot
.PHONY: all clean 

all : nerror_xsiny.$(FORMAT) nerror_xyz.$(FORMAT) plot_sequence

nerror_xsiny.$(FORMAT) : check.txt 
	$(ECHO) '\
	set term $(FORMAT); \
	set output "$@"; \
	set title "Error of f(x, y) = x sin(y) as a function of sample size"; \
	set tics out; \
	set key above ; \
	set xlabel "Sample size [N]"; \
	set ylabel "Error"; \
	plot \
	"$<" index 0 using 1:2 with lines title "MC plain - exact" \
	, "$<" index 0 using 1:3 with lines title "MC quasi - exact" \
	, "$<" index 0 using 1:4 with lines ls 1 dashtype 2 title "Estimated error on plain" \
	, "$<" index 0 using 1:5 with lines ls 2 dashtype 2 title "Estimated error on quasi" \
	' | $(PLOTTER)

nerror_xyz.$(FORMAT) : check.txt 
	$(ECHO) '\
	set term $(FORMAT); \
	set output "$@"; \
	set title "Error of f(x, y, z) = 8xyz as a function of sample size"; \
	set tics out; \
	set key above ; \
	set xlabel "Sample size [N]"; \
	set ylabel "Error"; \
	plot \
	"$<" index 1 using 1:2 with lines title "MC plain - exact" \
	, "$<" index 1 using 1:3 with lines title "MC quasi - exact" \
	, "$<" index 1 using 1:4 with lines ls 1 dashtype 2 title "Estimated error on plain" \
	, "$<" index 1 using 1:5 with lines ls 2 dashtype 2 title "Estimated error on quasi" \
	' | $(PLOTTER)

check.txt : main.py
	python3 $< > $@

plot_sequence : plot_sequences.py
	python3 $<

clean : 
	rm -f check.txt *.$(FORMAT) *.pdf
	@echo "cleaned up"
