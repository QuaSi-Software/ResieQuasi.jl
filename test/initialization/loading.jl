using Debugger
using Test
import ResieQuasi as Resie
using ResieQuasi.EnergySystems

include("../test_util.jl")

function test_load_from_dict()
    components_config = OrderedDict{String,Any}("TST_GRI_01" => Dict{String,Any}(
                                                                    "type" => "GridSupply",
                                                                    "output_refs" => ["TST_HP_01"],
                                                                    "medium" => "m_h_w_lt1",
                                                                ),
                                                "TST_GRI_02" => Dict{String,Any}(
                                                                    "type" => "GridSupply",
                                                                    "output_refs" => ["TST_HP_01"],
                                                                    "medium" => "m_e_ac_230v",
                                                                ),
                                                "TST_HP_01" => Dict{String,Any}(
                                                                   "type" => "HeatPump",
                                                                   "output_refs" => ["TST_BT_01"],
                                                                   "control_modules" => [Dict{String,Any}(
                                                                                             "name" => "storage_driven",
                                                                                             "high_threshold" => 0.5,
                                                                                             "low_threshold" => 0.1,
                                                                                             "storage_uac" => "TST_BT_01",
                                                                                         )],
                                                                   "power_th" => 20000,
                                                                   "cop_function" => "const:3.0",
                                                                   "power_losses_factor" => 1.0,
                                                                   "heat_losses_factor" => 1.0,
                                                               ),
                                                "TST_BT_01" => Dict{String,Any}(
                                                                   "type" => "BufferTank",
                                                                   "output_refs" => ["TST_DEM_01"],
                                                                   "medium" => "m_h_w_ht1",
                                                                   "model_type" => "ideally_stratified",
                                                                   "capacity" => 40000,
                                                                   "initial_load" => 0.5,
                                                               ),
                                                "TST_DEM_01" => Dict{String,Any}(
                                                                    "type" => "Demand",
                                                                    "output_refs" => [],
                                                                    "medium" => "m_h_w_ht1",
                                                                    "constant_demand" => 5000,
                                                                    "constant_temperature" => 60,
                                                                ))
    simulation_params = get_default_sim_params()
    components = Resie.load_components(components_config, simulation_params)
    @test length(keys(components)) == 5
    @test typeof(components["TST_BT_01"]) == Resie.EnergySystems.BufferTank
    @test components["TST_BT_01"].sys_function == Resie.EnergySystems.sf_storage
    @test components["TST_HP_01"].design_power_th == 20000
end

@testset "load_from_dict" begin
    test_load_from_dict()
end

@testset "simulation performance profiling setting" begin
    project_config = OrderedDict{String,Any}("io_settings" => Dict{String,Any}())
    io_settings = Resie.get_io_settings(project_config)
    @test io_settings["profile_simulation_performance"] == false
    @test io_settings["performance_plot_file_path"] == "./output/performance_profile.html"

    project_config["io_settings"]["profile_simulation_performance"] = true
    io_settings = Resie.get_io_settings(project_config)
    @test io_settings["profile_simulation_performance"] == true
end

@testset "simulation performance detail recording" begin
    sim_params = Dict{String,Any}(
        "performance_detail_timings" => Dict{Tuple{String,Symbol,Symbol},UInt64}(),
        "performance_detail_counts" => Dict{Tuple{String,Symbol,Symbol},Int}())

    Resie.record_performance_detail!(sim_params, "HeatPump A", :potential, :optimizer_total,
                                     UInt64(100), 1)
    Resie.record_performance_detail!(sim_params, "HeatPump A", :potential, :optimizer_total,
                                     UInt64(250), 2)

    key = ("HeatPump A", :potential, :optimizer_total)
    @test sim_params["performance_detail_timings"][key] == UInt64(350)
    @test sim_params["performance_detail_counts"][key] == 3

    unprofiled_params = Dict{String,Any}()
    @test Resie.record_performance_detail!(unprofiled_params, "HeatPump A", :process,
                                           :optimizer_total, UInt64(100)) === nothing
end

@testset "simulation performance plot" begin
    performance_timings = Dict{Symbol,UInt64}(:process => 3_000_000_000,
                                              :operation_reordering => 1_000_000_000)
    component_timings = Dict{String,UInt64}("Unit A" => 3_000_000_000)
    component_stage_timings = Dict{Tuple{String,Symbol},UInt64}(("Unit A", :process) => 3_000_000_000)

    mktempdir() do directory
        plot_path = joinpath(directory, "performance_profile.html")
        result_path = Resie.create_simulation_performance_plot(plot_path,
                                                               performance_timings,
                                                               component_timings,
                                                               component_stage_timings,
                                                               UInt64(4_000_000_000))
        @test result_path == plot_path
        @test isfile(plot_path)
        @test occursin("Unit A", read(plot_path, String))
        @test occursin("Simulation overhead", read(plot_path, String))
    end
end
