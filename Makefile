FLAGS = -Wall -std=gnu99
LDLIBS = -lm
CFLAGS += `gsl-config --cflags`
LDLIBS += `gsl-config --libs`
ECHO = /bin/echo
FORMAT = pdf
SCRIPT = log
PLOTTER = pyxplot

.PHONY: all plot.$(FORMAT) clean

all : plot.$(FORMAT)

n = 0
x_start = 0.1
x_end = 5
x_delta = 0.2

plot.$(FORMAT) : out.txt Makefile
	$(RM) $(SCRIPT)
	$(ECHO) '\
	set term $(FORMAT); \
	set output "$@"; \
	set title "Bessel function for n=$(n)"; \
	set tics out; \
	set grid; \
	set xlabel "x"; \
	set ylabel "$$J_n(x)$$"; \
	plot \
	"$<" index 0 with points title "Calculated" \
	, "$<" index 1 with lines title "Exact" \
	' | tee $(SCRIPT) | $(PLOTTER)

out.txt : jn
	#./$^ > $@
	./$< $(n) $(x_start) $(x_end) $(x_delta)  > $@

jn : jn.c
	cc $^ -o $@ $(CFLAGS) $(LDLIBS)

clean : 
	find ./ -type f -executable -delete
	$(RM) *.txt $(SCRIPT) plot.$(FORMAT) 
