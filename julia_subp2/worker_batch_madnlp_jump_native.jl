if !isdefined(Main, :__SUBP2_JULIA_ENV_ACTIVE__)
    import Pkg
    Pkg.activate(@__DIR__)
    const __SUBP2_JULIA_ENV_ACTIVE__ = true
end

using JSON3
using MadNLP
using LinearAlgebra

include("solve_batch_madnlp_jump_native.jl")

# -----------------------------------------------------------------------------
# 这个文件不是 SubP2 求解核心，而是 Julia 侧的 IPC worker 包装层。
#
# 当前主线关系是：
#   Python subprocess.Popen(...)
#     -> 启动这个 worker
#     -> 通过 stdin 发送一行 JSON 命令
#     -> worker 调 solve_batch_madnlp_jump_native.jl 里的函数
#     -> 通过 stdout 回一行 JSON 结果
#
# 如果你现在只是想看"正式主线必经路径"，最小阅读顺序是：
# 1. main()
# 2. handle_command(cmd)
# 3. action == "solve_batch" 且 haskey(cmd, :payload) 这一支
# 4. solve_batch_payload(...)   # 真正的 batch 求解在被 include 的主文件里
#
# 可以先跳过的内容：
# - action == "load_batch_only"           # 纯诊断 / 解析耗时测试
# - action == "prepare_batch"            # 预热 / 诊断，不是必须求解路径
# - solve_batch(..., in_path=...) 分支       # 文件路径兼容模式，主线默认不走
# -----------------------------------------------------------------------------

# worker 启动时只做一次 BLAS 线程配置。
# 目的是避免外层 Julia 线程和底层 BLAS 再次多线程叠加。
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

# worker 的命令分发中心。
# 它本身不做复杂求解，只负责把 JSON action 映射到对应函数。
function handle_command(cmd)
    action = String(get(cmd, :action, ""))
    # 主线会先发 ping，确认 worker 已起来，并读取线程/solver 配置。
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
    # 可先跳过：纯诊断分支。
    # 只测 JSON 文件读取/解析，不做求解。
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
    # 可先跳过：预热/诊断分支。
    # 用来提前建 template / cache，把首次建模成本和 solve 成本拆开。
    elseif action == "prepare_batch"
        max_iter = Int(get(cmd, :max_iter, 200))
        acceptable_tol = Float64(get(cmd, :acceptable_tol, 1e-4))
        acceptable_iter = Int(get(cmd, :acceptable_iter, 5))
        tol = Float64(get(cmd, :tol, 1e-8))
        # 当前主线更偏向直接传 payload，而不是只给 in_path。
        # 当前正式主线走这里：直接吃内存里的 payload。
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
            # 可先跳过：文件路径兼容模式。
            # 这条是给离线 JSON 文件调试用的，当前 Python 主线默认不走。
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
    # 正式主线最关键的分支。
    # Python 会把 stacked runtime payload 发到这里，再转给 solve_batch_payload(...)。
    elseif action == "solve_batch"
        max_iter = Int(get(cmd, :max_iter, 200))
        acceptable_tol = Float64(get(cmd, :acceptable_tol, 1e-4))
        acceptable_iter = Int(get(cmd, :acceptable_iter, 5))
        tol = Float64(get(cmd, :tol, 1e-8))
        runtime_metric_mode = Symbol(String(get(cmd, :runtime_metric_mode, "full")))
        result_format = Symbol(String(get(cmd, :result_format, "default")))
        # 当前正式主线走这里：直接吃内存里的 payload。
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
            # 可先跳过：文件路径兼容模式。
            # 这条是给离线 JSON 文件调试用的，当前 Python 主线默认不走。
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
    # 主线收尾时会发 shutdown，让 worker 正常退出。
    elseif action == "shutdown"
        return Dict("ok" => true, "action" => "shutdown")
    else
        return Dict("ok" => false, "error" => "unknown action: $(action)")
    end
end

# worker 主循环：
# - 阻塞等待 stdin 的一行 JSON
# - 调 handle_command
# - 把结果写回 stdout
#
# 这是一个同步 request/response 模型，不是异步消息队列。
function main()
    # 空闲时 worker 就阻塞在这里等 Python 发下一条命令。
    while !eof(stdin)
        line = try
            readline(stdin)
        catch
            break
        end
        isempty(strip(line)) && continue
        # 协议层统一兜底：任何异常都包装成 {ok:false, error:...} 回给 Python。
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

# 这个文件本身就是 worker 进程入口，所以直接执行 main()。
main()
