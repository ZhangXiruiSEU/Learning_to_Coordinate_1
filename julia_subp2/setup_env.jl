import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()

println("Julia SubP2 environment ready at ", @__DIR__)

