set term pdfcairo
set out "plot.A.pdf"
set key left
plot "out.A.txt" index 0 with errorbars title "data" \
, "out.A.txt" index 1 with lines title "fit" \
, "out.A.txt" index 2 with lines title "fit + err" \
, "out.A.txt" index 3 with lines title "fit - err"
