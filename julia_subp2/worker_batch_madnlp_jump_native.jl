if !isdefined(Main, :__SUBP2_JULIA_ENV_ACTIVE__)
    import Pkg
    Pkg.activate(@__DIR__)
    const __SUBP2_JULIA_ENV_ACTIVE__ = true
end

using JSON3
using MadNLP
using LinearAlgebra

include("solve_batch_madnlp_jump_native.jl")

function _configure_blas_threads()
    val = get(ENV, "JULIA_BLAS_THREADS", "1")
    try
        n = max(parse(Int, val), 1)
        BLAS.set_num_threads(n)
    catch
    end
    return BLAS.get_num_threads()
end

const _BLAS_THREADS = _configure_blas_threads()
const _BACKEND_CFG = _get_backend_config()

function handle_command(cmd)
    action = String(get(cmd, :action, ""))
    if action == "ping"
        return Dict(
            "ok" => true,
            "action" => "pong",
            "result" => Dict(
                "nthreads" => Threads.nthreads(),
                "blas_threads" => _BLAS_THREADS,
                "thread_schedule" => _thread_schedule_name(),
                "gpu_outer_mode" => _gpu_outer_mode(),
                "linear_solver" => _BACKEND_CFG.linear_solver_name,
                "kkt_system" => _BACKEND_CFG.kkt_system_name,
                "callback" => _BACKEND_CFG.callback_name,
            ),
        )
    elseif action == "load_batch_only"
        in_path = String(cmd[:in_path])
        t0 = time_ns()
        payload = JSON3.read(read(in_path, String))
        elapsed_ms = (time_ns() - t0) / 1.0e6
        steps = get(payload, :steps, [])
        return Dict(
            "ok" => true,
            "action" => action,
            "result" => Dict(
                "nsteps" => length(steps),
                "load_parse_ms" => elapsed_ms,
            ),
        )
    elseif action == "prepare_batch"
        max_iter = Int(get(cmd, :max_iter, 200))
        acceptable_tol = Float64(get(cmd, :acceptable_tol, 1e-4))
        acceptable_iter = Int(get(cmd, :acceptable_iter, 5))
        tol = Float64(get(cmd, :tol, 1e-8))
        result = if haskey(cmd, :payload)
            payload = cmd[:payload]
            redirect_stdout(devnull) do
                prepare_batch_payload!(
                    payload;
                    max_iter = max_iter,
                    acceptable_tol = acceptable_tol,
                    acceptable_iter = acceptable_iter,
                    tol = tol,
                )
            end
        else
            in_path = String(cmd[:in_path])
            payload = JSON3.read(read(in_path, String))
            redirect_stdout(devnull) do
                prepare_batch_payload!(
                    payload;
                    max_iter = max_iter,
                    acceptable_tol = acceptable_tol,
                    acceptable_iter = acceptable_iter,
                    tol = tol,
                )
            end
        end
        return Dict(
            "ok" => true,
            "action" => action,
            "result" => result,
        )
    elseif action == "solve_batch"
        max_iter = Int(get(cmd, :max_iter, 200))
        acceptable_tol = Float64(get(cmd, :acceptable_tol, 1e-4))
        acceptable_iter = Int(get(cmd, :acceptable_iter, 5))
        tol = Float64(get(cmd, :tol, 1e-8))
        runtime_metric_mode = Symbol(String(get(cmd, :runtime_metric_mode, "full")))
        result_format = Symbol(String(get(cmd, :result_format, "default")))
        result = if haskey(cmd, :payload)
            payload = cmd[:payload]
            redirect_stdout(devnull) do
                solve_batch_payload(
                    payload;
                    max_iter = max_iter,
                    acceptable_tol = acceptable_tol,
                    acceptable_iter = acceptable_iter,
                    tol = tol,
                    runtime_metric_mode = runtime_metric_mode,
                    result_format = result_format,
                )
            end
        else
            in_path = String(cmd[:in_path])
            redirect_stdout(devnull) do
                solve_batch(
                    in_path;
                    max_iter = max_iter,
                    acceptable_tol = acceptable_tol,
                    acceptable_iter = acceptable_iter,
                    tol = tol,
                    runtime_metric_mode = runtime_metric_mode,
                    result_format = result_format,
                )
            end
        end
        return Dict(
            "ok" => true,
            "action" => action,
            "result" => result,
        )
    elseif action == "shutdown"
        return Dict("ok" => true, "action" => "shutdown")
    else
        return Dict("ok" => false, "error" => "unknown action: $(action)")
    end
end

function main()
    while !eof(stdin)
        line = try
            readline(stdin)
        catch
            break
        end
        isempty(strip(line)) && continue
        response = try
            cmd = JSON3.read(line)
            handle_command(cmd)
        catch err
            Dict(
                "ok" => false,
                "error" => sprint(showerror, err),
            )
        end
        JSON3.write(stdout, response)
        write(stdout, '\n')
        flush(stdout)
        if get(response, "action", nothing) == "shutdown"
            break
        end
    end
end

main()
