#!/usr/bin/env gnuplot

set style line  1 lt 1 lw 1.0 pt 6 ps 1.0 lc rgb "black"
set style line  2 lt 1 lw 1.0 pt 4 ps 1.0 lc rgb "red"
set style line  3 lt 1 lw 1.0 pt 8 ps 1.0 lc rgb "blue"
set style line  4 lt 1 lw 1.0 pt 2 ps 1.0 lc rgb "forest-green"
set style line  5 lt 1 lw 1.0 pt 7 ps 1.0 lc rgb "orange"

set style line 10 lt 1 lw 1.0 lc rgb "black"
set style line 20 lt 3 lw 1.0 lc rgb "black"

#------------------ function to create the final PDF ------------------#

makepdf(fname) = sprintf("\
set output; \
set print \"| /bin/bash\"; \
print \"latex %s\"; \
print \"dvipdfm %s.dvi &>/dev/null\"; \
print \"rm -f %s.tex %s-inc.eps %s.dvi %s.log %s.aux\"; \
set print", fname, fname, fname, fname, fname, fname, fname)

#----------------------------- plot size ------------------------------#

mm    = 0.0393700787    # inch/mm
pt    = 0.996264009963  # pt/bp
plotw = 70*mm
ploth = 0.9*plotw

#-------------------------- terminal set-up ---------------------------#

set terminal epslatex color dashlength 2.0 font "" 8 \
    size plotw, ploth clip standalone lw 2.0 \
    header "\\usepackage[T1]{fontenc}\n\\usepackage{mathptmx}"

set border front ls 10 

set tics front
set mxtics 2
set mytics 2

#set tmargin 0.5
#set bmargin 1.5

pc_ref = 0.1992365

#----------------------------------------------------------------------#
#                              P_infinity                              #
#----------------------------------------------------------------------#

set output "p_inf.tex"

set xtics  format "%3.2f"
set ytics  format "%3.1f"

set title "$P_{\\infty}$ vs.\ site probability (different supercells)"
set xlabel "site probability $p$"
set ylabel "$P_{\\infty}(p)$" offset 1.0

set key bottom right spacing 1.3 samplen 1.0 reverse Left

set arrow 1 from pc_ref, 0.0 to pc_ref, 1.0 nohead ls 20

plot "./1x1x1/percol.out" u 1:3 w lp ls 1 t "$L =  4$",  \
     "./2x2x2/percol.out" u 1:3 w lp ls 2 t "$L =  8$",  \
     "./3x3x3/percol.out" u 1:3 w lp ls 3 t "$L = 12$", \
     "./4x4x4/percol.out" u 1:3 w lp ls 4 t "$L = 16$"

eval makepdf("p_inf")

#---------------------------- convergence -----------------------------#

set output "pc_conv.tex"

set xtics  format "%.0f"
set ytics  format "%.3f"

set title "convergence of $p_c$ with supercell"
set xlabel "$N_{\\textup{sites}}=L^3$"
set ylabel "$p_c$ as estimated as turning point of $P_{\\infty}(p)$" offset 1.0

set xtics (64, 512, 1728, 4096)

set arrow 1 from 64, pc_ref to 4096, pc_ref nohead ls 20

plot "pc-fit.dat" u ($1*$1*$1):2 w lp ls 1 t ""

eval makepdf("pc_conv")

#----------------------------------------------------------------------#
#                                 P_s                                  #
#----------------------------------------------------------------------#

set output "p_s.tex"

set xtics  format "%3.2f"
set ytics  format "%3.1f"

set title "$P_{s}$ vs.\ site probability (different supercells)"
set xlabel "site probability $p$"
set ylabel "$P_{s}(p)$" offset 1.0

set xtics auto

set key top left spacing 1.3 samplen 1.0 reverse Left

set arrow 1 from pc_ref, 0.0 to pc_ref, 1.0 nohead ls 20

plot "./1x1x1/percol.out" u 1:2 w lp ls 1 t "$L =  4$",  \
     "./2x2x2/percol.out" u 1:2 w lp ls 2 t "$L =  8$",  \
     "./3x3x3/percol.out" u 1:2 w lp ls 3 t "$L = 12$", \
     "./4x4x4/percol.out" u 1:2 w lp ls 4 t "$L = 16$"

eval makepdf("p_s")

#----------------------------------------------------------------------#
#                            susceptibility                            #
#----------------------------------------------------------------------#

set output "chi.tex"

set xtics  format "%.2f"
set ytics  format "%.1f"

set title "$\\chi$ vs.\ site probability (different supercells)"
set xlabel "site probability $p$"
set ylabel "$\\chi(p)$ (arbitrary units)" offset 1.0

set xtics auto

set key top right spacing 1.3 samplen 1.0 reverse Left

set arrow 1 from pc_ref, 0.0 to pc_ref, 1.2 nohead ls 20

plot "./1x1x1/percol.out" u 1:($4/(   30.0)) w lp ls 1 t "$L =  4$",  \
     "./2x2x2/percol.out" u 1:($4/( 2000.0)) w lp ls 2 t "$L =  8$",  \
     "./3x3x3/percol.out" u 1:($4/(18000.0)) w lp ls 3 t "$L = 12$", \
     "./4x4x4/percol.out" u 1:($4/(80000.0)) w lp ls 4 t "$L = 16$"

eval makepdf("chi")
