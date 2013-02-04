#!/usr/bin/env gnuplot

set style line  1 lt 1 lw 1.0 pt 6 ps 1.0 lc rgb "black"
set style line  2 lt 1 lw 1.0 pt 4 ps 1.0 lc rgb "red"
set style line  3 lt 1 lw 1.0 pt 8 ps 1.0 lc rgb "blue"
set style line  4 lt 1 lw 1.0 pt 2 ps 1.0 lc rgb "green"
set style line  5 lt 1 lw 1.0 pt 7 ps 1.0 lc rgb "orange"
set style line  6 lt 1 lw 1.0 pt 7 ps 1.0 lc rgb "brown"

set style line 11 lt 3 lw 1.5 pt 6 ps 1.0 lc rgb "black"
set style line 12 lt 3 lw 1.5 pt 4 ps 1.0 lc rgb "red"
set style line 13 lt 3 lw 1.5 pt 8 ps 1.0 lc rgb "blue"
set style line 14 lt 3 lw 1.5 pt 2 ps 1.0 lc rgb "green"
set style line 15 lt 3 lw 1.5 pt 7 ps 1.0 lc rgb "orange"

set style line 10 lt 1 lw 1.0 lc rgb "black"
set style line 20 lt 2 lw 1.0 lc rgb "black"

#------------------ function to create the final PDF ------------------#

makepdf(fname) = sprintf("\
set output; \
set print \"| /bin/bash\"; \
print \"latex %s\"; \
print \"dvipdfm %s.dvi &>/dev/null\"; \
print \"rm -f %s.tex %s-inc.eps %s.dvi %s.log %s.aux\"; \
set print", fname, fname, fname, fname, fname, fname, fname)

fc(x,y,rc) = (x<=rc) ? y : 1/0

#----------------------------- plot size ------------------------------#

mm    = 0.0393700787    # inch/mm
pt    = 0.996264009963  # pt/bp
plotw = 80*mm
ploth = 0.6*plotw

#-------------------------- terminal set-up ---------------------------#

set terminal epslatex color dashlength 2.0 font "" 8 \
    size plotw, ploth clip standalone lw 2.0 \
    header "\\usepackage[T1]{fontenc}\n\\usepackage{mathptmx}"

set border front ls 10 

set tics front
set mxtics 2
set mytics 2

set rmargin 2.0

#-------------------- critical site concentrations --------------------#

pc1   = 0.19993018 # 32x32x32
pc1c1 = 0.33734766 # 8x8x8c1

#------------------- fraction of percolating bonds --------------------#

name = "bonds"
set output name.".tex"

f_b(p) = (1.0-p)**2*(1.0-p**4)

set xrange [0:1]
set yrange [0:1]
set xtics format "%.1f"
set ytics format "%.1f"

set xlabel "transition metal concentration $p$"
set ylabel "fraction of percolating bonds" offset 0.5

set key top right reverse Left samplen 1.2 spacing 1.5 width -6.0

set arrow 1 from (1.0-pc1c1), 0.0 to (1.0-pc1c1), 0.58 nohead ls 1
set arrow 2 from (1.0-pc1), 0.0   to (1.0-pc1), 0.58   nohead ls 12

set samples 20
plot f_b(x)                              w p ls  1 t "$(1-p)^2(1-p^4)$", \
     "8x8x8c1/percol.bonds" u (1.0-$1):2 w l ls  1 t "Gerd's percolation rule", \
     "8x8x8/percol.bonds"   u (1.0-$1):2 w l ls 12 t "standard site percolation"
set samples 500

eval makepdf(name)

#------------------------- inaccessible sites -------------------------#

name = "inacc"
set output name.".tex"

set xtics format "%.1f"
set ytics format "%.1f"

set xlabel "transition metal concentration $p$"
set ylabel "fraction of inaccessible Li (rel.)" offset 0.5

set key top left reverse Left samplen 1.2 spacing 1.5

set arrow 1 from (1.0-pc1c1), 0.0 to (1.0-pc1c1), 1.0 nohead ls 1
set arrow 2 from (1.0-pc1), 0.0   to (1.0-pc1), 1.0   nohead ls 12

plot "8x8x8c1/percol.inacc" u (1.0-$1):2 w l ls  1 t "Gerd's percolation rule", \
     "6x6x6/percol.inacc"   u (1.0-$1):(fc((1.0-$1),$2,0.95)) w l ls 12 t "standard site percolation"

eval makepdf(name)

#------------------ inaccessible sites (convergence) ------------------#

name = "inacc-conv"
set output name.".tex"

set xrange [0.5:1.0]
set xtics format "%.1f"
set ytics format "%.1f"

set xlabel "transition metal concentration $p$"
set ylabel "fraction of inaccessible Li (rel.)" offset 0.5

set key top left reverse Left samplen 1.2 spacing 1.5

set arrow 1 from (1.0-pc1), 0.0 to (1.0-pc1), 1.0 nohead ls 1 front
unset arrow 2

plot "4x4x4/percol.inacc"    u (1.0-$1):(fc((1.0-$1),$2,0.95)) w l ls 1 t "4x4x4", \
     "8x8x8/percol.inacc"    u (1.0-$1):(fc((1.0-$1),$2,0.95)) w l ls 2 t "8x8x8", \
     "16x16x16/percol.inacc" u (1.0-$1):(fc((1.0-$1),$2,0.95)) w l ls 3 t "16x16x16"

eval makepdf(name)

#------------------------ wrapping probability ------------------------#

name = "wrap"
set output name.".tex"

set xrange [0.0:0.5]
set yrange [0:1.0]
set xtics format "%.1f"
set ytics format "%.1f"

set xlabel "concentration $p$"
set ylabel "wrapping probability" offset 0.5

set key top left reverse Left samplen 1.2 spacing 1.5

set arrow 1 from pc1, 0.0 to pc1, 1.0 nohead ls 10 front

plot "4x4x4/percol.wrap"    u 1:3 w l  ls 1 t "4x4x4", \
     "8x8x8/percol.wrap"    u 1:3 w l  ls 2 t "8x8x8", \
     "16x16x16/percol.wrap" u 1:3 w l  ls 3 t "16x16x16", \
     "32x32x32/percol.wrap" u 1:3 w l  ls 4 t "32x32x32"

eval makepdf(name)

