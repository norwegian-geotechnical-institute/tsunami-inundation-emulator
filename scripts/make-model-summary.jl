using NCDatasets, Flux, CUDA, Distributed, DataFrames, CSV, Serialization
using Statistics, Random, StatsBase
using ArgParse
import YAML
using DelimitedFiles: readdlm, writedlm
"""
copy to, and run from model folder. 
"""

s = ArgParseSettings()

@add_arg_table s begin
    "--scenarios"
        help = "Path to text file listing all scenarios, or train/test"
        arg_type = String
    "--eval_dir"
        help = "Store output in directory."
        arg_type = String
        default = pwd()
end

args = parse_args(s)

# Retrieve the values
dataset = args["scenarios"]
eval_dir = args["eval_dir"]
call_directory = pwd()

@info "dataset: $dataset, eval_dir: $eval_dir"

# Move to model directory.
addprocs(3; exeflags="--project")

@everywhere begin
    using Logging
    global_logger(SimpleLogger(stdout, Logging.Info))
    include("datareader.jl")
    include("model_config.jl")
    config = DataReader.parse_config("config.yml")

    # Set here
    config["gpu"] = false
    config["batch_size"] = 10
end

# Set constants.
const depth_classes = [(0.,0.2) (0.2,1.) (1.,3.) (3., 99.)]
const ct_mask = BitArray(readdlm("ct_mask.txt", '\t',Bool, '\n'))

if dataset == "train"
    dataset = config["train_data"]
    eval_dir = joinpath(config["rundir"], "evaluation", "train")
    #evaluate_model(dataset, eval_dir)
elseif dataset == "test"
    dataset = config["test_data"]
    eval_dir = joinpath(config["rundir"], "evaluation", "test")
    #evaluate_model(dataset, eval_dir)
end

function evaluate_model(dataset, eval_dir)
    # Load model
    model = ModelConfig.load(joinpath(config["rundir"],"$(config["model_name"]).jls"))
    Flux.reset!(model)

    # Create evaluation directory
    if !isdir(eval_dir)
        mkpath(eval_dir)
    end
    
    # Load batches on workers.
    @everywhere begin
    reader = DataReader.Reader(config)
    end
    batches = RemoteChannel(()->Channel(4))
    scenarios = DataReader.scenarios(dataset)
    
    for worker in workers()
        remote_do(reader, worker, scenarios, batches)
    end

    nr_of_batches = ceil(countlines(dataset)/config["batch_size"])

    # Create mask for test points from file test_points.txt
    selected_mask = falses(size(ct_mask))
    poi_df = CSV.File(joinpath(rundir, "test_points.txt"); delim='\t') |> DataFrame;
    for df_row in eachrow(poi_df)
        selected_mask[df_row.row, df_row.col] = true
    end
    
    # Make predictions, calculate error and write to file.
    l2(x) = mean(x.^2)^0.5
    l1(x) = mean(abs.(x))

    open(joinpath(eval_dir, "summary_results.txt"), "w") do io_summary
        open(joinpath(eval_dir, "point_predictions.txt"), "w") do io_selected
            open(joinpath(eval_dir, "summary_by_class.txt"), "w") do io_class_summary
                writedlm(io_summary, ["scenario" "l2_err" "l1_err" "l2_norm" "l1_norm"])
                writedlm(io_selected, permutedims(vcat(map(i -> "p$i", poi_df.point_nr), map(i -> "t$i", poi_df.point_nr))))
                writedlm(io_class_summary, ["scenario" "class" "K" "kappa" "nr_of_samples_aida" "mean_res" "std_res" "q_res" "nr_of_samples_res"])

                for batch_nr in 1:nr_of_batches
                    batch = take!(batches)

                    @info "Batch: $batch_nr"
                    # Target
                    x = batch.flow_depths[:,:,1,:]

                    # Predictions
                    hat_x = zeros(Float32, (config["dims"]...,config["batch_size"]))
                    hat_x[ct_mask,:] = relu(model(batch.etas) - batch.deformed_topographies[ct_mask,1,:])

                    preds = hat_x[selected_mask,:]
                    targets = x[selected_mask,:]

                    for idx in 1:config["batch_size"]
                        err = x[:,:,idx] - hat_x[:,:,idx]
                        writedlm(io_summary, [batch.scenario_names[idx] l2(err) l1(err) l2(x[:,:,idx]) l1(x[:,:,idx])])
                        writedlm(io_selected, permutedims(vcat(preds[:,idx], targets[:,idx])))
                        for i in 1:length(depth_classes)
                            K, kappa, nr_of_samples_aida = aida_stats(hat_x[:,:,idx], x[:,:,idx], depth_classes[i])
                            mean_res, std_res, q_res, nr_of_samples_res = residual_stats(hat_x[:,:,idx], x[:,:,idx], depth_classes[i])
                            writedlm(io_class_summary, 
                                     [batch.scenario_names[idx] i K kappa nr_of_samples_aida mean_res std_res q_res nr_of_samples_res]
                                    )
                        end
                    end
                end
            end
        end
    end
end

function aida_stats(pred, target,(alpha, beta))
    # Aidas numbers
    mask = alpha .< target .< beta .&& 1e-4 .< pred
    nr_of_samples = sum(mask)

    log_Ki = nr_of_samples > 0 ? log.(target[mask]./pred[mask]) : missing
    log_K = nr_of_samples > 0 ? mean(log_Ki) : missing
    log_kappa = nr_of_samples > 1 ? std(log_Ki) : missing

    return exp(log_K), exp(log_kappa), nr_of_samples
end

function residual_stats(pred, target,(alpha, beta))
    # Residual statistics
    mask = alpha .< target .< beta
    nr_of_samples = sum(mask)

    residuals = target[mask] .- pred[mask]
    mean_res = nr_of_samples > 0 ? mean(residuals) : missing
    std_res = nr_of_samples > 1 ? std(residuals) : missing
    q_res = nr_of_samples > 0 ? quantile(abs.(residuals), 0.95) : missing

    return mean_res, std_res, q_res, nr_of_samples
end

@info "model-path: $model_path, scenarios: $dataset, eval_dir: $eval_dir, call_directory: $call_directory"
@time evaluate_model(dataset, eval_dir)
