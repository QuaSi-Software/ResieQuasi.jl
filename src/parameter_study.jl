using Optim: Optim
using BlackBoxOptim: BlackBoxOptim
using Metaheuristics: Metaheuristics
using NLopt: NLopt
using NOMAD: NOMAD
using PolyChaos: PolyChaos
using LinearAlgebra
using Statistics
using Random

# structs to hold the simulation result in cash to avoid simulating the same parameter set multiple times
mutable struct ParameterEvaluationEntry
    completed::Base.Event
    result::Any
    failure::Any
end

mutable struct ParameterEvaluationCache
    lock::ReentrantLock
    entries::Dict{Tuple,ParameterEvaluationEntry}
end

ParameterEvaluationCache() = ParameterEvaluationCache(ReentrantLock(), Dict{Tuple,ParameterEvaluationEntry}())

"""
    get_or_run_parameter_set!(cache, key, run_simulation)

Return a cached simulation result or run the simulation if not cached.

Ensures thread-safe deduplication: concurrent requests for the same parameter
set share one simulation, while different parameter sets run in parallel.
Failed simulations are not cached, allowing subsequent retries.

# Arguments
- `cache::ParameterEvaluationCache`: Shared simulation result cache.
- `key::Tuple`: Tuple of canonical parameter values.
- `run_simulation::Function`: Function executing the simulation.

# Returns
- The cached or newly computed simulation result.
"""
function get_or_run_parameter_set!(cache::ParameterEvaluationCache,
                                   key::Tuple, run_simulation::Function)
    # hold the lock only to select the owner of this parameter set
    entry, is_owner = lock(cache.lock) do
        if haskey(cache.entries, key)
            return cache.entries[key], false
        end

        new_entry = ParameterEvaluationEntry(Base.Event(), nothing, nothing)
        cache.entries[key] = new_entry
        return new_entry, true
    end

    if !is_owner
        # event is persistent: a late waiter also sees an earlier notification
        wait(entry.completed)
        return lock(cache.lock) do
            entry.failure === nothing || throw(entry.failure)
            return entry.result
        end
    end

    try
        # no other run present, start new run
        result = run_simulation()
        lock(cache.lock) do
            entry.result = result
            # Do not retain failed runs. Already waiting callers still see this
            # attempt's result; subsequent calls may retry the simulation
            if get(result, "error", "") != ""
                delete!(cache.entries, key)
            end
            notify(entry.completed)
        end
        return result
    catch err
        lock(cache.lock) do
            entry.failure = err
            if get(cache.entries, key, nothing) === entry
                delete!(cache.entries, key)
            end
            # always release callers waiting for this simulation
            notify(entry.completed)
        end
        rethrow()
    end
end

"""
    create_parameter_variant(io_settings, sim_params, project_config, parameter_set)

Create a variant of the input file for the simulation run

# Arguments
- `io_settings::Dict{String,Any}`: IO settings
- `sim_params::Dict{String, Any}`: Simulation parameters
- `project_config::OrderedDict{String,Any}`: The project config read from the input file
- `parameter_set::Dict{String,Any}`: The parameters that get changed for the simulation run
# Returns
- `Tuple{OrderedDict{String,Any},String}`: Modified project config and output filename prefix
"""
function create_parameter_variant(io_settings::Dict{String,Any}, sim_params::Dict{String,Any},
                                  project_config::OrderedDict{String,Any},
                                  parameter_set::Dict{String,Any})::Tuple{OrderedDict{String,Any},String}
    # helper function to shorten parameter values for output file naming
    function compact_value(value)
        if abs(value) >= 100
            return string(round(Int, value))
        elseif abs(value) >= 10
            return string(round(value; digits=1))
        else
            return string(round(value; digits=2))
        end
    end

    # crete initial config for the current run
    cfg = deepcopy(project_config)

    # set up the parameters for this simulation variant
    run_name = ""
    for (key, value) in pairs(parameter_set)
        uac, param_key = split(key, " ")
        parameter_config = haskey(cfg["components"], uac) ? cfg["components"][uac] : cfg[uac]
        container, leaf_key = resolve_parameter_path(parameter_config, param_key)
        container[leaf_key] = value
        run_name *= uac * "_" * param_key * "_" * compact_value(value) * "_"
    end

    # rename outputs to clarify parameter values if outputfiles for each simulation should be generated
    for (key, value) in pairs(io_settings)
        if endswith(key, "file_path") && io_settings[key] != "nothing"
            dir, filename = splitdir(value)
            root, ext = splitext(filename)
            new_path = joinpath(dir, run_name * "_" * root * ext)
            cfg["io_settings"][key] = new_path
        end
    end

    return cfg, run_name * "_"
end

"""
    evaluate_parameter_set!(evaluated_parameter_sets, io_settings, sim_params,
                            parameter_study_results_path, project_config, parameter_values,
                            run_lock, output_lock, results_lock)

Evaluate one physical parameter set and store the simulation result. The returned objective is
compatible with optimisation algorithms and sensitivity calculations.

# Arguments
- `evaluated_parameter_sets::Vector{Any}`: Results of all completed parameter sets
- `io_settings::Dict{String,Any}`: IO settings
- `sim_params::Dict{String,Any}`: Simulation parameters
- `parameter_study_results_path::Union{String,Nothing}`: File path for parameter-study results.
  If `nothing`, the result is not written to the main parameter-study CSV.
- `project_config::OrderedDict{String,Any}`: The project config
- `parameter_values`: Physical parameter values proposed for the simulation run.
- `run_lock::ReentrantLock`: Lock for writing to current_runs
- `output_lock::ReentrantLock`: Lock for file at parameter_study_results_path
- `results_lock::ReentrantLock`: Lock for evaluated_parameter_sets
- `cancel_flag::Union{Nothing,Threads.Atomic{Bool}}`: Flag to pass STR+C down to all parallel runs
- `preparation_cache::Union{Nothing,PreparationCache}`: Optional shared cache for reusable simulation preparation data.
- `evaluation_cache::Union{Nothing,ParameterEvaluationCache}`: Optional thread-safe cache to prevent duplicate 
   simulations and reuse results.
# Returns
- `Union{Array{Float64},Float64}`: Objective value used by the configured study
"""
function evaluate_parameter_set!(evaluated_parameter_sets::Vector{Any}, io_settings::Dict{String,Any},
                                 sim_params::Dict{String,Any},
                                 parameter_study_results_path::Union{String,Nothing},
                                 project_config::OrderedDict{String,Any},
                                 parameter_values, run_lock::ReentrantLock,
                                 output_lock::ReentrantLock,
                                 results_lock::ReentrantLock;
                                 cancel_flag::Union{Nothing,Threads.Atomic{Bool}}=nothing,
                                 preparation_cache::Union{Nothing,PreparationCache}=nothing,
                                 evaluation_cache::Union{Nothing,ParameterEvaluationCache}=nothing)::Union{Array{Float64},
                                                                                                           Float64}
    parameter_study = sim_params["parameter_study"]
    runtime = parameter_study["runtime"]
    values = canonical_study_values(parameter_values, runtime)
    parameter_set = Dict{String,Any}(zip(runtime["parameter_keys"], values))

    # simulate the sample if not already stored in cash
    function simulate_once()
        run_ID = uuid4()
        results = run_simulation_sample(io_settings,
                                        sim_params,
                                        parameter_study_results_path,
                                        project_config,
                                        parameter_set,
                                        run_ID,
                                        run_lock,
                                        output_lock;
                                        suppress_all_output=parameter_study["disable_all_simulation_outputs"],
                                        cancel_flag=cancel_flag,
                                        preparation_cache=preparation_cache)
        return results
    end

    results = if evaluation_cache === nothing
        simulate_once()
    else
        get_or_run_parameter_set!(evaluation_cache, Tuple(values), simulate_once)
    end

    # add all simulation results to result container, even if they were taken from cache
    lock(results_lock) do
        push!(evaluated_parameter_sets, results)
    end

    return objective_for_parameter_study(results, parameter_study)
end

# Parameter-study result helpers

function objective_for_parameter_study(results::AbstractDict, parameter_study::Dict{String,Any})
    objective = results["objective"]

    if parameter_study["runtime"]["N_obj"] == 1
        return objective
    end

    # apply signs for each objective for multi-objective optimisation
    return Float64.(objective) .* parameter_study["runtime"]["objective_signs"]
end

# transform normalised optimisation parameters back to physical values
function to_physical_optim_values(sample_values, bounds::AbstractMatrix{<:Real})::Vector{Float64}
    values = sample_values isa Real ? [Float64(sample_values)] : vec(Float64.(sample_values))
    return bounds[:, 1] .+ values .* (bounds[:, 2] .- bounds[:, 1])
end

# transform physical parameters to normalised optimisation values
function to_normalised_optim_values(sample_values, bounds::AbstractMatrix{<:Real})
    values = sample_values isa Real ? [Float64(sample_values)] : vec(Float64.(sample_values))
    return (values .- bounds[:, 1]) ./ (bounds[:, 2] .- bounds[:, 1])
end

function best_objective_result(evaluated_parameter_sets::Vector{Any},
                               parameter_study::Dict{String,Any})
    parameter_study["runtime"]["N_obj"] == 1 || return nothing

    valid_results = filter(evaluated_parameter_sets) do result
        result isa AbstractDict || return false

        objective = get(result, "objective", nothing)
        return objective isa Real && isfinite(objective)
    end

    isempty(valid_results) && return nothing

    return valid_results[argmin(result["objective"] for result in valid_results)]
end

"""
    report_best_objective_result(evaluated_parameter_sets, parameter_study, workflow_name)

Log the best successfully evaluated single-objective parameter-study result.
Parameter values are reported in their physical units.

# Arguments
- `evaluated_parameter_sets::Vector{Any}`: Results of all completed parameter sets
- `parameter_study::Dict{String,Any}`: The parameter-study configuration
- `workflow_name::String`: The workflow name

"""
function report_best_objective_result(evaluated_parameter_sets::Vector{Any},
                                      parameter_study::Dict{String,Any},
                                      workflow_name::String)
    if isempty(evaluated_parameter_sets)
        @globalInfo "No $workflow_name result is available."
        return
    end

    if parameter_study["runtime"]["N_obj"] != 1
        # No single best solution can be reported for multi-objective parameter studies.
        @globalInfo "No single best solution can be reported for multi-objective parameter studies."
        return
    end

    best_result = best_objective_result(evaluated_parameter_sets, parameter_study)

    if isnothing(best_result)
        @globalInfo "No valid $workflow_name result was found."
        return
    end

    parameter_keys = parameter_study["runtime"]["parameter_keys"]
    parameter_width = max(length("Parameter"), maximum(length.(parameter_keys)))

    header = @sprintf("%-*s  %16s", parameter_width, "Parameter", "Value",)
    lines = [header, repeat("-", length(header))]

    for key in parameter_keys
        push!(lines, @sprintf("%-*s  %16.8g", parameter_width, key, Float64(best_result[key]),))
    end

    @globalInfo("Best $workflow_name result:\n" *
                "Objective: $(round(Float64(best_result["objective"]); sigdigits=8))\n\n" *
                join(lines, "\n") * "\n")
end

# Sensitivity calculations

"""
    calculate_global_sensitivity!(evaluate_physical_values, bounds, evaluated_parameter_sets, parameter_keys,
                                  parameter_study, cancel_parameter_study)

Calculate the global sensitivity indices with polynomial chaos expansion (PCE). A 
surrogate 3rd degree polynomial model is fit to the existing data. If the existing data
doesn't produce a well enough fit more data is generated in batches until RMSE is < 0.1 or 
the configured sensitivity run limit is hit.

# Arguments
- `evaluate_physical_values::Function`: Function to run if more datapoints are needed
- `bounds::Array{Float64}`: Bounds in which to analyse parameters
- `evaluated_parameter_sets::Vector{Any}`: Results of all completed parameter sets
- `parameter_keys::Vector{String}`: Names of all variable parameters
- `parameter_study::Dict{String,Any}`: Parameter-study configuration
- `cancel_parameter_study::Threads.Atomic{Bool}`: Shared cancellation flag
# Returns
- `Float64`: Total-order Sobol sensitivity index
- `Float64`: First-order Sobol sensitivity index
- `Float64`: Relative root mean square error for the surrogate model
- `Float64`: R^2 for the surrogate model
"""
function calculate_global_sensitivity!(evaluate_physical_values::Union{Nothing,Function},
                                       bounds::Array{Float64},
                                       evaluated_parameter_sets::Vector{Any},
                                       parameter_keys::Array{String},
                                       parameter_study::Dict{String,Any},
                                       cancel_parameter_study::Threads.Atomic{Bool})::Tuple{Vector{Float64},
                                                                                            Vector{Float64},
                                                                                            Float64,Float64}
    @globalInfo "Calculating global sensitivity..."

    d = size(bounds, 1)
    runtime = parameter_study["runtime"]
    parameter_types = runtime["parameter_types"]
    has_integer_parameters = any(T <: Integer for T in parameter_types)
    # Avoid polynomial degrees that cannot be identified from a small integer grid.
    integer_cardinalities = [Int(bounds[i, 2] - bounds[i, 1]) + 1
                             for i in 1:d if parameter_types[i] <: Integer]
    deg = isempty(integer_cardinalities) ? 3 : min(3, minimum(integer_cardinalities) - 1)
    op = PolyChaos.Uniform01OrthoPoly(deg; Nrec=5 * deg)
    mop = PolyChaos.MultiOrthoPoly(fill(op, d), deg)

    # [-1,1] -> physical
    function to_phys(x, lo, hi)
        lo + (x + 1) * (hi - lo) / 2
    end

    # physical -> [-1,1]
    function to_std(x, lo, hi)
        (x - lo) / (hi - lo) * 2 - 1
    end

    # Fit PCE coefficients by least squares and estimate out-of-sample error via
    # leave-one-out CV using the hat-matrix shortcut:
    function fit_surrogate(X_std::Matrix{Float64}, y::Vector{Float64}, mop::PolyChaos.MultiOrthoPoly)
        Phi = zeros(length(y), mop.dim)
        for i in 1:length(y)
            Phi[i, :] = PolyChaos.evaluate(X_std[i, :], mop)
        end
        coeffs = Phi \ y
        resid = y .- Phi * coeffs
        H = Phi * pinv(Phi' * Phi) * Phi'     # hat matrix, only needed for its diagonal
        leverage = diag(H)
        loocv_resid = resid ./ max.(1 .- leverage, eps(Float64))
        variance_scale = std(y)
        total_deviation = sum((y .- mean(y)) .^ 2)
        rel_rmse = variance_scale <= eps(Float64) ? Inf :
                   sqrt(mean(loocv_resid .^ 2)) / variance_scale
        r2 = total_deviation <= eps(Float64) ? 0.0 :
             1 - sum(loocv_resid .^ 2) / total_deviation

        return coeffs, rel_rmse, r2
    end

    # Take samples (standardized on [-1,1]). Only results inside the selected bounds
    # can be reused for global sensitivity.
    valid_results = filter(evaluated_parameter_sets) do result
        result isa AbstractDict || return false
        objective = get(result, "objective", nothing)
        objective isa Real && isfinite(objective) || return false

        for (i, key) in enumerate(parameter_keys)
            value = get(result, key, nothing)
            value isa Real && isfinite(value) || return false
            bounds[i, 1] <= value <= bounds[i, 2] || return false
        end
        return true
    end

    n_existing = length(valid_results)
    keys = vcat("objective", parameter_keys...)
    if n_existing > 0
        res_matrix = Float64[d[k] for d in valid_results, k in keys]
        X_phys = res_matrix[:, 2:end]
        y = vec(res_matrix[:, 1])
    else
        y = Float64[]
        X_phys = Array{Float64}(undef, 0, d)
    end

    minimum_fit_samples = mop.dim + 1
    if has_integer_parameters && length(integer_cardinalities) == d &&
       prod(big.(integer_cardinalities)) < minimum_fit_samples
        @error "Global sensitivity requires at least $minimum_fit_samples distinct parameter points " *
               "for this surrogate, but the selected integer bounds permit only " *
               "$(prod(big.(integer_cardinalities))). Increase the integer ranges."
        return Float64[], Float64[], 0.0, 0.0
    end
    if n_existing >= minimum_fit_samples
        X_std = hcat([to_std.(X_phys[:, i], bounds[i, 1], bounds[i, 2]) for i in 1:d]...)
        coeffs, rel_rmse, r2 = fit_surrogate(X_std, y, mop)
    else
        rel_rmse = 1.0
        coeffs = zeros(mop.dim)
        r2 = 0.0
    end

    target_rel_rmse = 0.1
    target_r2 = 0.9

    if evaluate_physical_values !== nothing
        print_message = true
        maximum_sensitivity_runs = parameter_study["runtime"]["sensitivity_max_runs"] + length(y)
        if length(y) < minimum_fit_samples && maximum_sensitivity_runs < minimum_fit_samples
            @error "Global sensitivity requires at least $minimum_fit_samples runs for " *
                   "$(length(parameter_keys)) parameters, but max_runs is $maximum_sensitivity_runs."
            return Float64[], Float64[], 0.0, 0.0
        end
        status_lock = ReentrantLock()
        seen_samples = Set(Tuple(X_phys[i, :]) for i in axes(X_phys, 1))
        try
            while (rel_rmse > target_rel_rmse || r2 < target_r2) && length(y) < maximum_sensitivity_runs
                if print_message
                    @globalInfo("Performing simulation runs for sensitivity analysis.\n" *
                                "Press Ctrl+C any time to stop creating additional runs and to " *
                                "calculate the sensitivity from the already completed runs.")
                    print_message = false
                end

                current_sample_count = length(y)
                if current_sample_count < mop.dim
                    n_new = max(mop.dim + 1 - current_sample_count, Threads.nthreads())
                else
                    n_new = Threads.nthreads()
                end

                # Do not exceed the configured sensitivity-run limit.
                remaining_runs = maximum_sensitivity_runs - current_sample_count
                n_new = min(n_new, remaining_runs)
                n_new > 0 || break

                # Draw distinct points from the actual input domain: continuous
                # uniform for floats, discrete uniform for integers. Avoid
                # replicating points after integer canonicalisation.
                new_points = Vector{Vector{Float64}}()
                attempts = 0
                while length(new_points) < n_new && attempts < max(200, 50 * n_new)
                    attempts += 1
                    point = Float64[]
                    for j in 1:d
                        lo, hi = bounds[j, 1], bounds[j, 2]
                        value = parameter_types[j] <: Integer ?
                                rand(ceil(Int, lo):floor(Int, hi)) :
                                lo + rand() * (hi - lo)
                        push!(point, Float64(value))
                    end
                    key = Tuple(point)
                    if !(key in seen_samples)
                        push!(seen_samples, key)
                        push!(new_points, point)
                    end
                end
                if isempty(new_points)
                    @warn "No additional distinct global-sensitivity sample is available within the selected bounds."
                    break
                end
                X_phys_new = permutedims(hcat(new_points...))
                n_new = length(new_points)
                y_new = zeros(Float64, n_new)
                completed_in_batch = Ref(0)
                Threads.@threads for i in 1:n_new
                    y_new[i] = evaluate_physical_values(X_phys_new[i, :])

                    lock(status_lock) do
                        completed_in_batch[] += 1
                        total_completed = current_sample_count + completed_in_batch[]
                        @globalInfo "Global sensitivity status: $total_completed of up to $maximum_sensitivity_runs runs completed."
                    end
                end

                # Create candidate arrays first. They are committed only
                # after the new surrogate fit has completed successfully.
                X_phys_candidate = vcat(X_phys, X_phys_new)
                y_candidate = vcat(y, y_new)
                X_std_candidate = hcat([to_std.(X_phys_candidate[:, i], bounds[i, 1], bounds[i, 2]) for i in 1:d]...)

                coeffs_candidate, rel_rmse_candidate, r2_candidate = fit_surrogate(X_std_candidate, y_candidate, mop)

                # Commit only a completely simulated and fitted batch.
                X_phys = X_phys_candidate
                y = y_candidate
                coeffs = coeffs_candidate
                rel_rmse = rel_rmse_candidate
                r2 = r2_candidate

                @globalInfo("Current quality: relative RMSE=$(round(rel_rmse; digits=4)) (goal <= $target_rel_rmse), " *
                            "R²=$(round(r2; digits=4)) (goal >= $target_r2).")
            end
        catch e
            if e isa InterruptException
                cancel_parameter_study[] = true
                @globalInfo("Additional sensitivity runs interrupted by Ctrl+C. " *
                            "Continuing sensitivity calculation with $(length(y)) completed runs.")
            else
                rethrow()
            end
        end
    end

    # Calculate Sobol indices from coefficients 
    total_var = sum(coeffs[2:end] .^ 2)
    if total_var <= eps(Float64)
        @error "Global sensitivity cannot be calculated because the surrogate objective " *
               "has no variance."
        return Float64[], Float64[], 0.0, 0.0
    end

    S_first = zeros(d)
    S_total = zeros(d)
    for i in 1:d, k in 2:mop.dim
        degs = mop.ind[k, :]
        if degs[i] != 0
            S_total[i] += coeffs[k]^2
            if all(j == i || degs[j] == 0 for j in 1:d)
                S_first[i] += coeffs[k]^2
            end
        end
    end
    S_first ./= total_var
    S_total ./= total_var

    if has_integer_parameters
        # Coefficient-squared Sobol indices assume continuous uniform inputs and
        # orthogonal polynomials. For integer dimensions, use pick-freeze Monte
        # Carlo integration of the fitted surrogate over the *discrete* uniform
        # input distribution instead. This still requires a good surrogate fit.
        n_mc = 4096
        function draw_inputs()
            samples = rand(n_mc, d)
            for j in 1:d
                lo, hi = bounds[j, 1], bounds[j, 2]
                if parameter_types[j] <: Integer
                    samples[:, j] = Float64.(rand(ceil(Int, lo):floor(Int, hi), n_mc))
                else
                    samples[:, j] = lo .+ samples[:, j] .* (hi - lo)
                end
            end
            return samples
        end
        function predict_surrogate(samples)
            prediction = Vector{Float64}(undef, size(samples, 1))
            for row in axes(samples, 1)
                standardized = Float64[to_std(samples[row, j], bounds[j, 1], bounds[j, 2]) for j in 1:d]
                prediction[row] = dot(coeffs, PolyChaos.evaluate(standardized, mop))
            end
            return prediction
        end
        A = draw_inputs()
        B = draw_inputs()
        fA = predict_surrogate(A)
        fB = predict_surrogate(B)
        variance = var(vcat(fA, fB))
        if variance <= eps(Float64)
            @error "Mixed discrete/continuous global sensitivity has zero surrogate variance."
            return Float64[], Float64[], rel_rmse, r2
        end
        for j in 1:d
            AB = copy(A)
            AB[:, j] = B[:, j]
            fAB = predict_surrogate(AB)
            S_first[j] = mean(fB .* (fAB .- fA)) / variance
            S_total[j] = mean((fA .- fAB) .^ 2) / (2 * variance)
        end
        @globalInfo "Global sensitivity with integer parameters: Sobol indices estimated from the " *
                    "surrogate using discrete-uniform integer inputs. Check the reported surrogate fit quality."
    end

    parameter_width = max(length("Parameter"), maximum(length.(parameter_keys)))
    header = @sprintf("%-*s  %12s  %12s", parameter_width, "Parameter", "S_first", "S_total",)
    lines = [header, repeat("-", length(header))]

    for (key, first_order, total_order) in zip(parameter_keys, S_first, S_total)
        push!(lines, @sprintf("%-*s  %12.3f  %12.3f", parameter_width, key, first_order, total_order,))
    end

    @globalInfo("Global sensitivity results:\n\n" *
                join(lines, "\n") *
                "\n\nSurrogate quality: RMSE=$(round(rel_rmse; digits=3)), " *
                "R²=$(round(r2; digits=3))\n" *
                "Sobol indices depend on the selected parameter bounds.\n",)

    return S_total, S_first, rel_rmse, r2
end

function find_evaluated_parameter_set(evaluated_parameter_sets::Vector{Any},
                                      parameter_keys::Vector{String},
                                      parameter_values::AbstractVector{<:Real})
    for result in evaluated_parameter_sets
        result isa AbstractDict || continue
        objective = get(result, "objective", nothing)
        objective isa Real && isfinite(objective) || continue
        all(haskey(result, key) &&
                result[key] isa Real &&
                isapprox(Float64(result[key]), Float64(value); rtol=1e-10, atol=1e-12)
            for (key, value) in zip(parameter_keys, parameter_values)) || continue
        return result
    end
    return nothing
end

"""
    calculate_local_sensitivity!(evaluate_physical_values, evaluated_parameter_sets, parameter_study)

Calculate local one-at-a-time sensitivities around a configured start point or the best
single-objective optimisation result. Explicit sensitivity_lower and sensitivity_upper
values can be configured per parameter. Otherwise, the configured relative variation is
applied in both directions. No optimisation bounds are applied to local sensitivity points.

Existing simulation results are reused if all parameter values match the requested point.

# Arguments
- `evaluate_physical_values::Function`: Function evaluating physical parameter values.
- `evaluated_parameter_sets::Vector{Any}`: Results of all completed parameter-study runs.
- `parameter_study::Dict{String,Any}`: Parameter-study configuration.
# Returns
- `Vector{Dict{String,Any}}`: Local sensitivity results for each parameter.
"""
function calculate_local_sensitivity!(evaluate_physical_values::Function,
                                      evaluated_parameter_sets::Vector{Any},
                                      parameter_study::Dict{String,Any})::Vector{Dict{String,Any}}
    @globalInfo "Calculating local sensitivity..."

    sensitivity_results = Vector{Dict{String,Any}}()
    runtime = parameter_study["runtime"]
    parameter_keys = runtime["parameter_keys"]
    parameter_types = runtime["parameter_types"]

    # determine reference values
    if parameter_study["sensitivity_analysis"]["local_reference"] == "best_result"
        best_result = best_objective_result(evaluated_parameter_sets,
                                            parameter_study)

        if isnothing(best_result)
            @error "No valid optimisation result is available as reference for local sensitivity."
            return sensitivity_results
        end

        reference_values = Float64[best_result[key] for key in parameter_keys]
    else
        reference_values = Float64.(parameter_study["runtime"]["start_values"])
    end

    reference_values = Float64.(canonical_study_values(reference_values, runtime))

    # prepare all valid local-sensitivity points before starting parallel runs
    parameter_points = NamedTuple[]

    for (i, key) in enumerate(parameter_keys)
        lower_value = parameter_study["runtime"]["sensitivity_lower_values"][i]
        upper_value = parameter_study["runtime"]["sensitivity_upper_values"][i]

        if !(isfinite(lower_value) && isfinite(upper_value))
            delta = abs(reference_values[i]) * parameter_study["sensitivity_analysis"]["local_variation"]
            if delta == 0.0
                @error "Local sensitivity for parameter $key cannot use a relative variation " *
                       "because its reference value is zero. Configure sensitivity_lower and sensitivity_upper explicitly."
                continue
            elseif parameter_types[i] <: Integer
                # Small integers may have a requested +/-10% step below one unit.
                # Use the closest feasible symmetric integer step, with at least 1.
                delta = Float64(max(1, round(Int, delta)))
            end

            lower_value = reference_values[i] - delta
            upper_value = reference_values[i] + delta
            if parameter_types[i] <: Integer && reference_values[i] != 0.0
                realised = delta / abs(reference_values[i])
                requested = parameter_study["sensitivity_analysis"]["local_variation"]
                if !isapprox(realised, requested; atol=1e-10)
                    @globalInfo "Local sensitivity for integer parameter $key: requested +/-$(100 * requested)% " *
                                "becomes +/-$(round(100 * realised; digits=2))% (one or more whole-number steps). " *
                                "The plots and CSV show the actual parameter changes."
                end
            end
        end

        if !(lower_value < reference_values[i] < upper_value)
            @error "Local sensitivity values for parameter $key must satisfy sensitivity_lower < reference < sensitivity_upper."
            continue
        end

        lower_values = copy(reference_values)
        upper_values = copy(reference_values)

        lower_values[i] = lower_value
        upper_values[i] = upper_value

        push!(parameter_points,
              (key=key,
               reference_value=reference_values[i],
               lower_value=lower_value,
               upper_value=upper_value,
               lower_values=lower_values,
               upper_values=upper_values))
    end

    if isempty(parameter_points)
        return sensitivity_results
    end

    # determine which points still require simulation
    simulation_points = Vector{Vector{Float64}}()

    queued_points = Set{Tuple}()
    function queue_simulation_point!(parameter_values::Vector{Float64})::Nothing
        actual_values = Float64.(canonical_study_values(parameter_values, runtime))
        existing_result = find_evaluated_parameter_set(evaluated_parameter_sets,
                                                       parameter_keys,
                                                       actual_values)

        point_key = Tuple(actual_values)
        if isnothing(existing_result) && !(point_key in queued_points)
            push!(queued_points, point_key)
            push!(simulation_points, actual_values)
        end

        return nothing
    end

    queue_simulation_point!(reference_values)

    for point in parameter_points
        queue_simulation_point!(point.lower_values)
        queue_simulation_point!(point.upper_values)
    end

    number_of_evaluation_points = length(simulation_points)
    completed_points = Ref(0)
    status_lock = ReentrantLock()

    # evaluate all missing points in parallel
    Threads.@threads for run_index in eachindex(simulation_points)
        evaluate_physical_values(simulation_points[run_index])

        lock(status_lock) do
            completed_points[] += 1

            @globalInfo "Local sensitivity status: $(completed_points[]) of " *
                        "$number_of_evaluation_points runs completed."
        end
    end

    # all parallel evaluations have completed; result lookup is serial again
    reference_result = find_evaluated_parameter_set(evaluated_parameter_sets,
                                                    parameter_keys,
                                                    reference_values)

    # prepare results for output
    if isnothing(reference_result)
        @error "The local sensitivity reference result is missing."
        return sensitivity_results
    end

    reference_objective = Float64(reference_result["objective"])

    for point in parameter_points
        lower_result = find_evaluated_parameter_set(evaluated_parameter_sets,
                                                    parameter_keys,
                                                    point.lower_values)

        upper_result = find_evaluated_parameter_set(evaluated_parameter_sets,
                                                    parameter_keys,
                                                    point.upper_values)

        if isnothing(lower_result) || isnothing(upper_result)
            @error "Local sensitivity results for parameter $(point.key) are incomplete."
            continue
        end

        lower_objective = Float64(lower_result["objective"])
        upper_objective = Float64(upper_result["objective"])

        lower_change = lower_objective - reference_objective
        upper_change = upper_objective - reference_objective
        lower_change_relative = reference_objective == 0.0 ? NaN :
                                lower_change / abs(reference_objective)
        upper_change_relative = reference_objective == 0.0 ? NaN :
                                upper_change / abs(reference_objective)
        gradient = (upper_objective - lower_objective) / (point.upper_value - point.lower_value)
        elasticity = point.reference_value == 0.0 || reference_objective == 0.0 ?
                     NaN :
                     gradient * point.reference_value / reference_objective

        push!(sensitivity_results,
              Dict{String,Any}(
                  "parameter" => point.key,
                  "lower_value" => point.lower_value,
                  "reference_value" => point.reference_value,
                  "upper_value" => point.upper_value,
                  "lower_objective" => lower_objective,
                  "reference_objective" => reference_objective,
                  "upper_objective" => upper_objective,
                  "lower_change" => lower_change,
                  "upper_change" => upper_change,
                  "lower_change_relative" => lower_change_relative,
                  "upper_change_relative" => upper_change_relative,
                  "gradient" => gradient,
                  "elasticity" => elasticity,
              ))
    end

    if isempty(sensitivity_results)
        return sensitivity_results
    end

    parameter_width = max(length("Parameter"), maximum(length.(parameter_keys)))

    header = @sprintf("%-*s  %12s  %24s  %12s  %24s  %10s",
                      parameter_width,
                      "Parameter",
                      "Lower",
                      "Δ objective (Δ %)",
                      "Upper",
                      "Δ objective (Δ %)",
                      "Elasticity",)

    lines = [header, repeat("-", length(header))]

    for result in sensitivity_results
        lower_change = @sprintf("%.7g (%+.3f %%)", result["lower_change"], 100 * result["lower_change_relative"],)
        upper_change = @sprintf("%.7g (%+.3f %%)", result["upper_change"], 100 * result["upper_change_relative"],)

        push!(lines,
              @sprintf("%-*s  %12.7g  %24s  %12.7g  %24s  %10.4g",
                       parameter_width,
                       result["parameter"],
                       result["lower_value"],
                       lower_change,
                       result["upper_value"],
                       upper_change,
                       result["elasticity"],))
    end

    @globalInfo("Local sensitivity results:\n" *
                "Reference objective: $(round(reference_objective; sigdigits=8))\n\n" *
                join(lines, "\n") * "\n",)

    return sensitivity_results
end

# Parameter-study stages

"""
    run_parameter_variation!(evaluate_physical_values, parameter_study, cancel_parameter_study)

Evaluate all configured parameter combinations.
"""
function run_parameter_variation!(evaluate_physical_values::Function,
                                  parameter_study::Dict{String,Any},
                                  cancel_parameter_study::Threads.Atomic{Bool})
    try
        # handling status
        variation_start_time = now()
        completed_runs = Atomic{Int}(0)
        max_runs = length(parameter_study["runtime"]["iterator"])
        worker_count = min(Threads.nthreads(), max_runs)

        # run parameter sets on multi threads
        @threads for parameter_values in collect(parameter_study["runtime"]["iterator"])
            if cancel_parameter_study[]
                continue
            end

            try
                run_start_time = now()
                float_parameter_values = parameter_values isa Real ? Float64(parameter_values) :
                                         Float64.(collect(parameter_values))
                evaluate_physical_values(float_parameter_values)

                # handle logging
                runtime_seconds = round(Int, seconds(now() - run_start_time))
                runtime_minutes, runtime_remaining_seconds = divrem(runtime_seconds, 60)
                completed = atomic_add!(completed_runs, 1) + 1
                elapsed_seconds = max(1, round(Int, seconds(now() - variation_start_time)))
                if completed < worker_count
                    eta_text = "calculating..."
                else
                    results_per_second = completed / elapsed_seconds
                    eta_seconds = round(Int, (max_runs - completed) / results_per_second)
                    eta_minutes, eta_remaining_seconds = divrem(max(eta_seconds, 0), 60)
                    eta_text = "$eta_minutes min $(lpad(eta_remaining_seconds, 2, '0')) s"
                end

                @globalInfo "[$completed/$max_runs] → completed in $runtime_minutes min " *
                            "$(lpad(runtime_remaining_seconds, 2, '0')) s. ETA: $eta_text"
            catch e
                if e isa InterruptException
                    cancel_parameter_study[] = true
                    continue
                else
                    rethrow()
                end
            end
        end
    catch e
        if e isa InterruptException
            # Handles Ctrl+C delivered to the task coordinating @threads.
            cancel_parameter_study[] = true
        else
            rethrow()
        end
    end
end

"""
    run_optimisation!(evaluate_physical_values, evaluated_parameter_sets, parameter_study,
                      cancel_parameter_study)

Run the configured optimisation and optional refinement stage.
"""
function run_optimisation!(evaluate_physical_values::Function,
                           evaluated_parameter_sets::Vector{Any},
                           parameter_study::Dict{String,Any},
                           cancel_parameter_study::Threads.Atomic{Bool})
    # evaluate_normalised_values converts normalised parameter values to physical values
    # for the simulation run.
    evaluate_normalised_values = function (parameter_values)
        physical_values = to_physical_optim_values(parameter_values, parameter_study["runtime"]["parameter_bounds"])
        return evaluate_physical_values(canonical_study_values(physical_values,
                                                               parameter_study["runtime"];
                                                               enforce_bounds=true))
    end

    # handle Logging for all algorithms
    progress_lock = ReentrantLock()
    progress_evaluations = Ref(0)
    progress_best = Ref(Inf)
    progress_every = 1

    # handles the progress logging of evaluate_normalised_values
    evaluate_with_progress = function (parameter_values)
        cancel_parameter_study[] && throw(InterruptException())
        result = evaluate_normalised_values(parameter_values)
        cancel_parameter_study[] && throw(InterruptException())

        lock(progress_lock) do
            progress_evaluations[] += 1

            best_text = if parameter_study["runtime"]["N_obj"] == 1
                value = result isa Real ? Float64(result) : Float64(first(result))
                progress_best[] = min(progress_best[], value)
                string(round(progress_best[]; sigdigits=8))
            else
                "multi-objective"
            end

            if progress_evaluations[] == 1 ||
               progress_evaluations[] % progress_every == 0
                @globalInfo("Optimisation progress: evaluations=$(progress_evaluations[]), best=$best_text",)
            end
        end

        return result
    end

    # run main optimisation
    try
        @globalInfo "Starting simulations for optimisation..."
        run_optimiser_backend!(parameter_study, evaluate_with_progress, cancel_parameter_study)
    catch e
        if e isa InterruptException
            cancel_parameter_study[] = true
        else
            rethrow()
        end
    end

    # run optional refinement optimisation
    if !cancel_parameter_study[]
        try
            refinement_optimiser = create_refinement_optimiser(parameter_study,
                                                               parameter_study["optimisation"]["refinement"],
                                                               evaluated_parameter_sets)
            if !isnothing(refinement_optimiser)
                @globalInfo "Start refinement simulations..."
                run_optimiser_backend!(refinement_optimiser,
                                       evaluate_with_progress,
                                       cancel_parameter_study)
            end
        catch e
            if e isa InterruptException
                cancel_parameter_study[] = true
            else
                rethrow()
            end
        end
    end
end

"""
    run_global_sensitivity!(evaluate_physical_values, evaluated_parameter_sets,
                            parameter_study, cancel_parameter_study)

Run the configured global sensitivity analysis and report its runtime.
"""
function run_global_sensitivity!(evaluate_physical_values::Function,
                                 evaluated_parameter_sets::Vector{Any},
                                 parameter_study::Dict{String,Any},
                                 cancel_parameter_study::Threads.Atomic{Bool},
                                 io_settings::Dict{String,Any},
                                 sim_params::Dict{String,Any})::Bool
    sensitivity_start_time = now()
    results_before_sensitivity = length(evaluated_parameter_sets)
    S_total = Float64[]
    S_first = Float64[]
    rel_rmse = NaN
    r2 = NaN
    calculation_completed = false

    try
        S_total, S_first, rel_rmse, r2 = calculate_global_sensitivity!(evaluate_physical_values,
                                                                       parameter_study["runtime"]["parameter_bounds"][:,
                                                                                                                      1:2],
                                                                       evaluated_parameter_sets,
                                                                       parameter_study["runtime"]["parameter_keys"],
                                                                       parameter_study,
                                                                       cancel_parameter_study)
        calculation_completed = true
    catch e
        if e isa InterruptException
            cancel_parameter_study[] = true
        else
            rethrow()
        end
    end

    parameter_keys = parameter_study["runtime"]["parameter_keys"]
    bounds = parameter_study["runtime"]["parameter_bounds"][:, 1:2]
    if calculation_completed &&
       length(S_total) == length(parameter_keys) &&
       length(S_first) == length(parameter_keys)
        output_lines = String[]

        if io_settings["output_parameter_study_csv"]
            csv_path = parameter_study_csv_path(sim_params,
                                                io_settings,
                                                "global_sensitivity")
            write_global_sensitivity_csv(csv_path,
                                         parameter_keys,
                                         bounds,
                                         S_total,
                                         S_first,
                                         rel_rmse,
                                         r2)
            push!(output_lines, "  CSV: $csv_path")
        end

        plot_path = create_global_sensitivity_plot(parameter_keys,
                                                   S_total,
                                                   S_first,
                                                   rel_rmse,
                                                   r2,
                                                   io_settings,
                                                   sim_params)
        isempty(plot_path) || push!(output_lines, "  Plot: $plot_path")

        if !isempty(output_lines)
            @globalInfo "Global sensitivity outputs:\n" * join(output_lines, "\n")
        end
    elseif calculation_completed
        @error "Global sensitivity outputs were not created because the result dimensions do not match the parameters."
    end

    sensitivity_runtime_seconds = round(Int, seconds(now() - sensitivity_start_time))
    sensitivity_runtime_minutes, sensitivity_runtime_remaining_seconds = divrem(sensitivity_runtime_seconds, 60)
    sensitivity_runs = length(evaluated_parameter_sets) - results_before_sensitivity

    if cancel_parameter_study[]
        @globalInfo "Global sensitivity analysis interrupted after $sensitivity_runtime_minutes min " *
                    "$(lpad(sensitivity_runtime_remaining_seconds, 2, '0')) s. " *
                    "$sensitivity_runs additional runs completed."
        return false
    end

    @globalInfo "Global sensitivity analysis completed in $sensitivity_runtime_minutes min " *
                "$(lpad(sensitivity_runtime_remaining_seconds, 2, '0')) s. " *
                "$sensitivity_runs additional runs completed."
    return true
end

"""
    run_local_sensitivity!(evaluate_physical_values, evaluated_parameter_sets,
                           parameter_study, cancel_parameter_study)

Run the configured local sensitivity analysis and report its runtime.
"""
function run_local_sensitivity!(evaluate_physical_values::Function,
                                evaluated_parameter_sets::Vector{Any},
                                parameter_study::Dict{String,Any},
                                cancel_parameter_study::Threads.Atomic{Bool},
                                io_settings::Dict{String,Any},
                                sim_params::Dict{String,Any})::Bool
    sensitivity_start_time = now()
    results_before_sensitivity = length(evaluated_parameter_sets)
    sensitivity_results = Vector{Dict{String,Any}}()

    try
        sensitivity_results = calculate_local_sensitivity!(evaluate_physical_values,
                                                           evaluated_parameter_sets,
                                                           parameter_study)
    catch e
        if e isa InterruptException
            cancel_parameter_study[] = true
        else
            rethrow()
        end
    end

    # write results to file and create plots
    if !isempty(sensitivity_results)
        output_lines = String[]

        if io_settings["output_parameter_study_csv"]
            csv_path = parameter_study_csv_path(sim_params,
                                                io_settings,
                                                "local_sensitivity")
            write_local_sensitivity_csv(csv_path,
                                        sensitivity_results)
            push!(output_lines, "  CSV: $csv_path")
        end

        response_plot_path = create_local_sensitivity_response_overview_plot(sensitivity_results,
                                                                             io_settings,
                                                                             sim_params)
        isempty(response_plot_path) || push!(output_lines, "  Response overview plot: $response_plot_path")

        response_trends_plot_path = create_local_sensitivity_response_trends_plot(sensitivity_results,
                                                                                  io_settings,
                                                                                  sim_params)
        isempty(response_trends_plot_path) || push!(output_lines, "  Response trends plot: $response_trends_plot_path")

        if !isempty(output_lines)
            @globalInfo "Local sensitivity outputs:\n" * join(output_lines, "\n")
        end
    end

    sensitivity_runtime_seconds = round(Int, seconds(now() - sensitivity_start_time))
    sensitivity_runtime_minutes, sensitivity_runtime_remaining_seconds = divrem(sensitivity_runtime_seconds, 60)
    sensitivity_runs = length(evaluated_parameter_sets) - results_before_sensitivity

    if cancel_parameter_study[]
        @globalInfo "Local sensitivity analysis interrupted after $sensitivity_runtime_minutes min " *
                    "$(lpad(sensitivity_runtime_remaining_seconds, 2, '0')) s. " *
                    "$sensitivity_runs additional runs completed."
        return false
    end

    @globalInfo "Local sensitivity analysis completed in $sensitivity_runtime_minutes min " *
                "$(lpad(sensitivity_runtime_remaining_seconds, 2, '0')) s. " *
                "$sensitivity_runs additional runs completed."
    return true
end

function write_parameter_study_results(parameter_study_results_path::String,
                                       evaluated_parameter_sets::Vector{Any})
    open(prepare_output_path(parameter_study_results_path), "w") do file_handle
        # write header
        header = join(collect(keys(evaluated_parameter_sets[1])), ';') * "\n"
        write(file_handle, header)

        # write results
        for results in evaluated_parameter_sets
            row = join(collect(values(results)), ';') * "\n"
            row = replace(row, '.' => ',')
            write(file_handle, row)
        end
    end
end

"""
    perform_parameter_study(io_settings, sim_params, project_config)

Run the configured parameter variation, optimisation and sensitivity analyses and collect the
resulting simulation outputs. Sensitivity analyses are executed after the primary study and
reuse already completed simulation results where possible. They can also run standalone.

# Arguments
- `io_settings::Dict{String,Any}`: IO settings used for simulation output and result writing.
- `sim_params::Dict{String,Any}`: Simulation and parameter-study parameters.
- `project_config::OrderedDict{String,Any}`: Base project configuration used to generate
  simulation variants.
- `preparation_cache::Union{Nothing,PreparationCache}`: Optional shared cache for reusable 
  simulation preparation data.

# Returns
- `Bool`: Flag if the parameter study was successful (true) or not (false).
- `Vector{Any}`: Main parameter-study results.
- `Vector{Any}`: Results generated exclusively for local sensitivity analysis.
"""
function perform_parameter_study(io_settings::Dict{String,Any},
                                 sim_params::Dict{String,Any},
                                 project_config::OrderedDict{String,Any};
                                 preparation_cache::Union{Nothing,PreparationCache}=nothing)::Tuple{Bool,Vector{Any},
                                                                                                    Vector{Any}}
    # establish overarching locks for parallelization
    run_lock = ReentrantLock()
    output_lock = ReentrantLock()
    results_lock = ReentrantLock()

    parameter_study_results_path = parameter_study_csv_path(sim_params, io_settings, "all_results")
    open(prepare_output_path(parameter_study_results_path), "w") do file_handle
    end

    parameter_study = sim_params["parameter_study"]

    # Cache complete parameter tuples only for studies involving integers.
    # A continuous parameter with a different value always gives a new key.
    evaluation_cache = any(T <: Integer for T in parameter_study["runtime"]["parameter_types"]) ?
                       ParameterEvaluationCache() : nothing

    # calculate inputs that should be cached
    if preparation_cache !== nothing
        @globalInfo "Preparing reusable data for repeated simulation runs..."
        warmup_run_ID = uuid4()
        prepare_inputs(project_config, warmup_run_ID; preparation_cache=preparation_cache)
    end

    # handle interruption via STR+C for parallel parameter-study runs
    cancel_parameter_study = Threads.Atomic{Bool}(false)

    # prepare result vectors
    evaluated_parameter_sets = Vector{Any}()
    local_sensitivity_evaluated_parameter_sets = Vector{Any}()

    # Generic function using physical simulation parameter values. Results from the primary
    # study and global sensitivity are appended to evaluated_parameter_sets.
    evaluate_physical_values = function (parameter_values)
        if cancel_parameter_study[]
            throw(InterruptException())
        end

        return evaluate_parameter_set!(evaluated_parameter_sets,
                                       io_settings,
                                       sim_params,
                                       parameter_study_results_path,
                                       project_config,
                                       parameter_values,
                                       run_lock,
                                       output_lock,
                                       results_lock;
                                       cancel_flag=cancel_parameter_study,
                                       preparation_cache=preparation_cache,
                                       evaluation_cache=evaluation_cache)
    end

    @globalInfo "---- Simulation setup completed. ----"
    @globalInfo "---- Starting simulations on $(Threads.nthreads()) Threads. ----"

    main_start_time = now()
    local_reference_values = nothing

    if parameter_study["runtime"]["run_primary_study"]
        primary_start_time = now()
        workflow_name = parameter_study["parameter_variation"]["run_parameter_variation"] ?
                        "Parameter variation" : "Optimisation"

        if parameter_study["parameter_variation"]["run_parameter_variation"]
            @globalInfo "Starting simulations for parameter variation study..."
            run_parameter_variation!(evaluate_physical_values,
                                     parameter_study,
                                     cancel_parameter_study)
        else
            run_optimisation!(evaluate_physical_values,
                              evaluated_parameter_sets,
                              parameter_study,
                              cancel_parameter_study)
        end

        primary_runtime_seconds = round(Int, seconds(now() - primary_start_time))
        primary_runtime_minutes, primary_runtime_remaining_seconds = divrem(primary_runtime_seconds, 60)
        primary_run_count = length(evaluated_parameter_sets)

        if cancel_parameter_study[]
            # use snapshot to avoid interference from threads that may still be running
            final_results = lock(results_lock) do
                copy(evaluated_parameter_sets)
            end
            primary_run_count = length(final_results)
            @globalInfo "$workflow_name interrupted by user after $primary_runtime_minutes min " *
                        "$(lpad(primary_runtime_remaining_seconds, 2, '0')) s. " *
                        "$primary_run_count runs completed. Recovering intermediate results..."

            # detect and output best simulation result, independent of any backend-specific results
            report_best_objective_result(final_results, parameter_study, workflow_name)
            return false, final_results, local_sensitivity_evaluated_parameter_sets
        end

        @globalInfo "$workflow_name completed in $primary_runtime_minutes min " *
                    "$(lpad(primary_runtime_remaining_seconds, 2, '0')) s. " *
                    "$primary_run_count runs completed."

        # detect and output best simulation result, independent of any backend-specific results
        report_best_objective_result(evaluated_parameter_sets, parameter_study, workflow_name)
    end

    # Sensitivity analyses are calculated after parameter variation or optimisation so that
    # completed simulation results can be reused. They can also run without a primary study.
    if parameter_study["sensitivity_analysis"]["run_global_sensitivity"] &&
       !run_global_sensitivity!(evaluate_physical_values,
                                evaluated_parameter_sets,
                                parameter_study,
                                cancel_parameter_study,
                                io_settings,
                                sim_params)
        return false, evaluated_parameter_sets, local_sensitivity_evaluated_parameter_sets
    end

    if parameter_study["sensitivity_analysis"]["run_local_sensitivity"]
        # Local sensitivity may intentionally evaluate points outside the optimisation bounds.
        # Use a separate working vector so these points can reuse existing results without
        # becoming part of the main parameter-study result set.
        local_working_results = copy(evaluated_parameter_sets)
        existing_result_count = length(local_working_results)

        evaluate_local_physical_values = function (parameter_values)
            if cancel_parameter_study[]
                throw(InterruptException())
            end

            return evaluate_parameter_set!(local_working_results,
                                           io_settings,
                                           sim_params,
                                           nothing,
                                           project_config,
                                           parameter_values,
                                           run_lock,
                                           output_lock,
                                           results_lock;
                                           cancel_flag=cancel_parameter_study,
                                           preparation_cache=preparation_cache)
        end

        local_success = run_local_sensitivity!(evaluate_local_physical_values,
                                               local_working_results,
                                               parameter_study,
                                               cancel_parameter_study,
                                               io_settings,
                                               sim_params)

        local_sensitivity_evaluated_parameter_sets = lock(results_lock) do
            if length(local_working_results) > existing_result_count
                copy(local_working_results[(existing_result_count + 1):end])
            else
                Vector{Any}()
            end
        end

        if !local_success
            return false, evaluated_parameter_sets, local_sensitivity_evaluated_parameter_sets
        end
    end

    overall_runtime_seconds = round(Int, seconds(now() - main_start_time))
    overall_runtime_minutes, overall_runtime_remaining_seconds = divrem(overall_runtime_seconds, 60)
    @globalInfo "---- Parameter study finished in $overall_runtime_minutes min " *
                "$(lpad(overall_runtime_remaining_seconds, 2, '0')) s. " *
                "$(length(evaluated_parameter_sets) + length(local_sensitivity_evaluated_parameter_sets)) total runs completed. ----"

    # write results to parameter-study result file if not written continuously
    if !io_settings["write_parameter_study_csv_continuously"] && !isempty(evaluated_parameter_sets)
        write_parameter_study_results(parameter_study_results_path, evaluated_parameter_sets)
    end

    return true, evaluated_parameter_sets, local_sensitivity_evaluated_parameter_sets
end

# Optimisation backends

function create_refinement_optimiser(optimiser::Dict{String,Any},
                                     refinement_config::Union{Nothing,AbstractDict},
                                     stage_results::Vector{Any})
    if isnothing(refinement_config)
        return nothing
    end

    if optimiser["runtime"]["objective_function_name"] == "multi-objective"
        @globalInfo "No refinement will be done, as this is not possible for multi-objective optimisation."
        return nothing
    end

    best_result = best_objective_result(stage_results, optimiser)
    if isnothing(best_result)
        return nothing
    end

    physical_start = Float64[best_result[key] for key in optimiser["runtime"]["parameter_keys"]]
    normalised_start = to_normalised_optim_values(physical_start, optimiser["runtime"]["parameter_bounds"])

    return load_refinement_optimiser(optimiser, refinement_config, normalised_start)
end

function run_optimiser_backend!(optimiser::Dict{String,Any}, f_progress::Function,
                                cancel_parameter_study::Threads.Atomic{Bool})
    if optimiser["optimisation"]["type"] == "Optim"
        return Optim.optimize(f_progress, optimiser["runtime"]["args"]...)
    elseif optimiser["optimisation"]["type"] == "BlackBoxOptim"
        if optimiser["runtime"]["N_obj"] == 1
            f_wrap = f_progress
        else
            f_wrap(x) = Tuple(f_progress(x))
        end

        return BlackBoxOptim.bboptimize(f_wrap, optimiser["runtime"]["args"]...; optimiser["runtime"]["kwargs"]...)
    elseif optimiser["optimisation"]["type"] == "Metaheuristics"
        # Scalar evaluation: used by MOEA/D-DE and other non-batch algorithms.
        function f_metaheuristics(sample_values::AbstractVector)
            result = f_progress(sample_values)

            if optimiser["runtime"]["N_obj"] == 1
                return Float64(result)
            end
            # Metaheuristics multi-objective scalar callback format:
            # (objectives, inequality constraints, equality constraints)
            return vec(Float64.(result)), [0.0], [0.0]
        end

        # Batch evaluation: used only by algorithms supporting parallel_evaluation.
        function f_metaheuristics(sample_values::AbstractMatrix)
            N_samples = size(sample_values, 1)
            objectives = zeros(N_samples, optimiser["runtime"]["N_obj"])

            Threads.@threads for i in 1:N_samples
                if cancel_parameter_study[]
                    continue
                end

                try
                    result = f_progress(view(sample_values, i, :))

                    if optimiser["runtime"]["N_obj"] == 1
                        objectives[i, 1] = Float64(result)
                    else
                        objectives[i, :] = vec(Float64.(result))
                    end
                catch e
                    if e isa InterruptException
                        cancel_parameter_study[] = true
                    else
                        rethrow()
                    end
                end
            end

            if cancel_parameter_study[]
                throw(InterruptException())
            end

            if optimiser["runtime"]["N_obj"] == 1
                return vec(objectives)
            else
                return objectives, zeros(N_samples, 1), zeros(N_samples, 1)
            end
        end

        return Metaheuristics.optimize(f_metaheuristics, optimiser["runtime"]["args"]...)
    elseif optimiser["optimisation"]["type"] == "NLopt"
        f_nlopt = function (sample_values, gradient)
            return f_progress(sample_values)
        end
        NLopt.min_objective!(optimiser["runtime"]["args"][1], f_nlopt)

        return NLopt.optimize(optimiser["runtime"]["args"]...)
    elseif optimiser["optimisation"]["type"] == "NOMAD"
        f_nomad = function (sample_values)
            result = f_progress(sample_values)
            outputs = result isa Real ? [Float64(result)] : Float64.(collect(result))
            success = all(isfinite, outputs)
            return success, true, outputs
        end

        prob = NOMAD.NomadProblem(optimiser["runtime"]["args"][1:(end - 1)]...,
                                  f_nomad;
                                  optimiser["runtime"]["kwargs"]...)

        return NOMAD.solve(prob, optimiser["runtime"]["args"][4])
    end

    @error "Unsupported optimiser type: $(optimiser["optimisation"]["type"])"
end
