FORMAT = pdf
.PHONY: all a_check.txt b_check.txt amain.py bmain.py aplot.$(FORMAT) clean 

all : aplot.$(FORMAT)

aplot.$(FORMAT): a_check.txt
	echo '\
	set out "$@";\
	set term $(FORMAT); \
	set xlabel "x";\
	set ylabel "y";\
	set title "Least-Squares fit of /F(x)=c_0x+c_1+c_2*x to data" ;\
	set tics out; \
	set key above; \
	set datafile separator "\t"; \
	plot "$<" index 0 with errorbars title "Data"\
	,"$<" index 1 using 1:2 with lines title "F(x)"\
	,"$<" index 1 using 1:3 with lines title "F(x) + dF(x)"\
	,"$<" index 1 using 1:4 with lines title "F(x) - dF(x)"\
	'|gnuplot

a_check.txt : amain.py
	python3 $< > $@

b_check.txt : bmain.py
	python3 $< > $@

clean : 
	rm -f a_check.txt b_check.txt aplot.$(FORMAT)
	@echo "cleaned up"
