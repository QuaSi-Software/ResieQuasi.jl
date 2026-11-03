module PlotUtils

using Dates
using OteraEngine
using PlotlyJS
using JSON
using ..Profiles: add_ignoring_leap_days

"""
Custom exception `InputError` used to signify that an input was not correctly set up,
outside the allowed range, etc.
Call with `throw(InputError("msg"))` or `throw(InputError())`.
"""
struct InputError <: Exception
    msg::Union{AbstractString,Nothing}
end
InputError() = InputError(nothing)

"""
    add_leap_day_values(values::AbstractArray, start_date::DateTime, 
                        time_step_second::UInt, end_date::DateTime)::AbstractArray

Add values for leap_days so x0 and dx can be used in plots together with rangebreaks.
Added values are the same as the next real value.

# Arguments
- `values::AbstractArray`: Data array that leap day values get added too
- `start_date::DateTime`: start date of the values
- `time_step_second::UInt`: time step of the values in seconds
- `end_date::DateTime`: end date of the values

# Returns
- `AbstractArray`: values data array with added leap day values
"""
function add_leap_day_values(values::AbstractArray, start_date::DateTime,
                             time_step_second::UInt, end_date::DateTime)::AbstractArray
    dates = collect(start_date:Second(time_step_second):end_date)
    leap_mask = (month.(dates) .== 2) .& (day.(dates) .== 29)
    y_with_leap = fill(NaN, length(dates), size(values, 2))
    y_with_leap[.!leap_mask, :] .= values
    for i in reverse(findall(leap_mask))
        y_with_leap[i, :] .= y_with_leap[i + 1, :]
    end
    return y_with_leap
end

"""
    downsampling_for_plots(values::AbstractArray, time_dim::Int64; 
                           max_n_values::Union{Int64, Nothing}=50_000_000, 
                           decimation_factor::Union{Int64, Nothing}=nothing)::Tuple{AbstractArray,Int64}

Downsamples an array 'values' on dimension 'time_dim' to reduce file size for plotting.
If decimation_factor is given, it always takes priority of max_n_values.
At least one of max_n_values or decimation_factor has to be given.

# Arguments
- `values::AbstractArray`: Data array to be downsampled
- `time_dim::Int64`: Dimension of array that carries time steps & is going to be downsampled
- `max_n_values::Union{Int64,Nothing}=50_000_000`: Maximum number of values in the array after 
                                                 downsampling
- `decimation_factor::Union{Int64,Nothing}=nothing`: Factor to be downsampled by

# Returns
- `AbstractArray`: Downsampled data array
- `Int64`: decimation_factor
"""
function downsampling_for_plots(values::AbstractArray, time_dim::Int64;
                                max_n_values::Union{Int64,Nothing}=50_000_000,
                                decimation_factor::Union{Int64,Nothing}=nothing)::Tuple{AbstractArray,Int64}
    if max_n_values <= 0
        @error "max_n_values must be > 0 for downsampling"
    end
    n_vals = prod(size(values))
    if isnothing(decimation_factor) && !isnothing(max_n_values)
        decimation_factor = ceil(Int, n_vals / max_n_values)
    elseif isnothing(decimation_factor) && isnothing(max_n_values)
        @error "At least one of max_n_values or decimation_factor has to be given for" *
               "function downsampling_for_plots()"
    end

    if decimation_factor > 1
        values = collect(selectdim(values, time_dim, 1:decimation_factor:size(values, time_dim)))
        @info "Values for plot got downsampled by factor $decimation_factor to reduce " *
              "file size."
    end
    return values, decimation_factor
end

"""
    save_html_slider(output_path::AbstractString, values_y_x_t::AbstractArray, 
                    base_plot::PlotlyJS.Plot, sim_params::Dict{String,Any}; 
                    digits::Int=2, render_time_ms::Number=40, 
                    play_time_ms::Number=20_000, max_n_values=50_000_000)

Create custom HTML file for 3D slider graph to improve performance.

Order of values MUST be y,x,t with reference to PloltyJS surface plot.
base_plot defines the layout and type of plots (example in SeasonalThermalStorage and 
GeothermalHeatCollector)

# Arguments
- `output_path::AbstractString`: Target path of the generated HTML file.
- `values_y_x_t::AbstractArray`: 3D data array with dimensions `(y, x, t) with reference to 
                                 PloltyJS surface plot`.
- `base_plot::PlotlyBase.Plot`: Base PlotlyJS plot defining the visual style, layout, 
                                and trace type.
- `sim_params::Dict{String,Any}`: Simulation parameters
- `digits::Int=2`: Number of decimal places used for rounding data.
- `render_time_ms::Number=40`: Approximate render time per frame in milliseconds.
- `play_time_ms::Number=20_000`: Target total playback duration for the slider animation.
"""
function save_html_slider(output_path::AbstractString, values_y_x_t::AbstractArray,
                          base_plot::PlotlyBase.Plot, sim_params::Dict{String,Any};
                          digits::Int=2, render_time_ms::Number=40,
                          play_time_ms::Number=20_000)
    values_y_x_t, decimation_factor = downsampling_for_plots(values_y_x_t, 3)
    ny, nx, nt = size(values_y_x_t)

    jumped_frames = ceil(Int, (nt * render_time_ms) / play_time_ms)
    values_int = round.(Int, vec(values_y_x_t) .* 10 ^ digits)

    time_step = sim_params["time_step_seconds"] * decimation_factor
    timestamps = Array{String}(undef, nt)
    for i in 1:nt
        timestamps[i] = string(add_ignoring_leap_days(sim_params["start_date_output"],
                                                      Dates.Second((i - 1) * time_step)))
    end
    template_path = joinpath(@__DIR__, "html_templates", "slider_plot_templates",
                             "3D_slider_plot_template.html")

    timestamps = replace.(timestamps, 'T' => ' ')
    replacements = Dict{Symbol,Any}(
        :PLOTLY_CDN_URL => "https://cdn.plot.ly/plotly-latest.min.js",
        #:PLOTLY_CDN_URL => PlotlyJS.PlotlyBase.cdn_url()
        :MAX_SLIDER => nt - 1,
        :VALUES_INT => join(values_int, ","),
        :TIMESTAMPS => JSON.json(timestamps),
        :N_X => nx,
        :N_Y => ny,
        :N_T => nt,
        :DIGITS => digits,
        :RENDER_TIME_MS => render_time_ms,
        :JUMPED_FRAMES => jumped_frames,
        :BASE_PLOT => JSON.json(base_plot),
    )

    template = OteraEngine.Template(template_path)(; init=replacements)
    write(output_path, template)
end

"""
    html_plot_insertion(file_path::String, template_name::String, 
                        replacements::Dict{String,Any}, insertion_location::String)::String

Takes a separate html insertion file for interactive plots and processes it.
"""
function html_plot_insertion(file_path::String, template_name::String,
                             replacements::Dict{Symbol,Any},
                             insertion_location::String)
    insertion_path = joinpath(@__DIR__, "html_templates",
                              "parameter_study_plots_insertions", template_name)
    insertion = OteraEngine.Template(insertion_path)(; init=replacements)

    html = read(file_path, String)

    if insertion_location == "body" && occursin("</body>", html)
        html = replace(html, "</body>" => insertion * "\n</body>"; count=1)
    elseif insertion_location == "head" && occursin("</head>", html)
        html = replace(html, "</head>" => insertion * "\n</head>"; count=1)
    else
        @error "insertion_location for html_plot_insertion has to be one of 'head' or 'body'"
    end

    write(file_path, html)

    # insertion = read(insertion_path, String)

    #TODO this check is not done by OteraEngine
    # for (name, value) in replacements
    #     placeholder = "{{$name}}"
    #     if !occursin(placeholder, insertion)
    #         @error "Missing placeholder $placeholder in parameter-study plot insertion $file_name."
    #     end
    #     insertion = replace(insertion, placeholder => value)
    # end

    # unresolved_placeholder = match(r"\{\{[A-Z0-9_]+\}\}", insertion)
    # if !isnothing(unresolved_placeholder)
    #     @error "Unresolved placeholder $(unresolved_placeholder.match) in parameter-study plot insertion $file_name."
    # end
    # return insertion
end

export html_plot_insertion, save_html_slider, downsampling_for_plots, add_leap_day_values
end
