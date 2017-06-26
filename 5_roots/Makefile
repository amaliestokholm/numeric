FORMAT = pdf
.PHONY: all a_check.txt c_check.txt amain.py cmain.py aplot.$(FORMAT) cplot.$(FORMAT) clean 

all : a_check.txt b_check.txt 

aplot.$(FORMAT): ab_check.txt
	echo '\
	set out "$@";\
	set term $(FORMAT); \
	set xlabel "x";\
	set ylabel "y";\
	set title "Least-Squares fit of F(x)=c_0/x+c_1+c_2x to data" ;\
	set tics out; \
	set key above; \
	set datafile separator "\t"; \
	plot "$<" index 0 with errorbars title "Data"\
	,"$<" index 1 using 1:2 with lines title "F(x)"\
	,"$<" index 1 using 1:3 with lines title "F(x) + error"\
	,"$<" index 1 using 1:4 with lines title "F(x) - error"\
	'|gnuplot

cplot.$(FORMAT): c_check.txt
	echo '\
	set out "$@";\
	set term $(FORMAT); \
	set xlabel "x";\
	set ylabel "y";\
	set title "Least-Squares fit of F(x)=c_0/x+c_1+c_2x to data using singular decompostion" ;\
	set tics out; \
	set key above; \
	set datafile separator "\t"; \
	plot "$<" index 0 with errorbars title "Data"\
	,"$<" index 1 using 1:2 with lines title "F(x)"\
	,"$<" index 1 using 1:3 with lines title "F(x) + error"\
	,"$<" index 1 using 1:4 with lines title "F(x) - error"\
	'|gnuplot

a_check.txt : amain.py
	python3 $< > $@

b_check.txt : bmain.py
	python3 $< > $@

clean : 
	rm -f a_check.txt c_check.txt aplot.$(FORMAT) cplot.$(FORMAT)
	@echo "cleaned up"
