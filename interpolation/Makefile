ECHO = /bin/echo
FORMAT = pdf
PLOTTER = gnuplot

.PHONY: all a_plot.$(FORMAT) b_plot.$(FORMAT) clean

all : a_plot.$(FORMAT) b_plot.$(FORMAT) 

a_plot.$(FORMAT) : a_out.txt 
	$(RM) $(SCRIPT)
	$(ECHO) '\
	set term $(FORMAT); \
	set output "$@"; \
	set title "Test of linear interpolation"; \
	set tics out; \
	set xlabel "x"; \
	set ylabel "cos(x)"; \
	plot \
	"$<" using 1:2  with lines title "Linear interpolation" \
	, "$<" using 1:3 with lines title "Integral" \
	, "$<" using 4:5 with points title "Data" \
	' | tee $(SCRIPT) | $(PLOTTER)

b_plot.$(FORMAT) : b_out.txt 
	$(RM) $(SCRIPT)
	$(ECHO) '\
	set term $(FORMAT); \
	set output "$@"; \
	set title "Test of quadratic interpolation"; \
	set tics out; \
	set xlabel "x"; \
	set ylabel "cos(x)"; \
	plot \
	"$<" using 1:2  with lines  title "Quadratic interpolation" \
	, "$<" using 1:3 with lines title "Integral" \
	, "$<" using 1:4 with lines title "Derivative" \
	, "$<" using 5:6 with points title "Data" \
	' | tee $(SCRIPT) | $(PLOTTER)


a_out.txt : amain.py
	python3 $< > $@

b_out.txt : bmain.py
	python3 $< > $@

clean : 
	rm -f *.txt a_plot.$(FORMAT) b_plot.$(FORMAT) 
	@echo "cleaned up"
