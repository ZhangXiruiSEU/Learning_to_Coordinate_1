import Pkg
Pkg.activate(dirname(@__DIR__))

using CUDA
using ExaModels
using JSON3
using MadNLP

println("Julia version: ", VERSION)
println("CUDA.functional(): ", CUDA.functional())
println("Loaded packages: ExaModels, JSON3, MadNLP")
