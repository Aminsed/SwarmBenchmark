set xlabel "X"
set ylabel "Y"
set title "Particle Swarm Optimization"

set xrange [-5.5:5.5]
set yrange [-5.5:5.5]

plot "particles.txt" using 1:2 with points pt 7 ps 0.5 title "Particles", \
     "best_position.txt" using 1:2 with points pt 7 ps 2 lc rgb "red" title "Best Position"

pause -1 "Press any key to continue..."
