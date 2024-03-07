set xlabel "X"
set ylabel "Y"
set title "Particle Swarm Optimization"

set xrange [-5.5:5.5]
set yrange [-5.5:5.5]

iterations = ITERATE_MAX  # Set the number of iterations

do for [i=0:iterations-1] {
    set title sprintf("Iteration %d", i)
    plot sprintf("particles_%d.txt", i) using 1:2 with points pt 7 ps 0.5 title "Particles", \
         "best_position.txt" using 1:2 with points pt 7 ps 2 lc rgb "red" title "Best Position"
    pause 0.1
}

pause -1 "Press any key to continue..."
